import json
import os
import time
from getpass import getpass
from pathlib import Path

import requests
from requests import exceptions as requests_exceptions


# 第一步：设置本地目录
base_data_dir = Path("/data/yuzhang_fei/PEMS")
base_data_dir.mkdir(parents=True, exist_ok=True)

print(f"✓ 数据将保存到根目录: {base_data_dir}")


# 第二步：输入 PeMS 账号
USERNAME = os.environ.get("PEMS_USERNAME") or input("PeMS 用户名: ").strip()
PASSWORD = os.environ.get("PEMS_PASSWORD") or getpass("PeMS 密码: ")
print("✓ 账号信息已读取")


def build_year_root_dir(district_id, year):
    return base_data_dir / f"PEMSD{district_id}_{year}"


def build_data_dir(district_id, year, data_type):
    return build_year_root_dir(district_id, year) / f"d{district_id:02d}_{data_type}"


# 第三步：初始化下载器
class PeMSDownloader:
    def __init__(self, username, password):
        self.session = requests.Session()
        self.base_url = "https://pems.dot.ca.gov/"
        self.username = username
        self.password = password
        self.connect_timeout = 20
        self.read_timeout = 60
        self.download_timeout = 300
        self.max_retries = 3
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": self.base_url,
        })

    def _request(self, method, url, timeout=None, **kwargs):
        request_timeout = timeout or (self.connect_timeout, self.read_timeout)
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.session.request(method, url, timeout=request_timeout, **kwargs)
            except (requests_exceptions.ConnectTimeout, requests_exceptions.ReadTimeout) as exc:
                last_error = exc
                print(f"  网络超时，第 {attempt}/{self.max_retries} 次重试: {url}")
                if attempt < self.max_retries:
                    time.sleep(attempt)
            except requests_exceptions.RequestException as exc:
                last_error = exc
                break

        if isinstance(last_error, requests_exceptions.ConnectTimeout):
            raise RuntimeError(
                "连接 PeMS 超时。当前机器可能无法访问 pems.dot.ca.gov，"
                "请先检查网络、代理、VPN 或防火墙策略。"
            ) from last_error
        if isinstance(last_error, requests_exceptions.ReadTimeout):
            raise RuntimeError(
                "PeMS 响应超时。站点可能较慢，或当前网络不稳定。"
            ) from last_error
        if last_error is not None:
            raise RuntimeError(f"请求 PeMS 失败: {last_error}") from last_error

        raise RuntimeError("请求 PeMS 失败，未获得响应")

    def _looks_like_login_page(self, text):
        lowered = (text or "").lower()
        markers = [
            "welcome to pems",
            "forgot your password",
            "username",
            "login",
        ]
        return sum(marker in lowered for marker in markers) >= 2

    def login(self):
        """登录 PeMS"""
        print("正在登录 PeMS...")
        try:
            self._request("GET", self.base_url)
            response = self._request(
                "POST",
                self.base_url,
                data={
                    "username": self.username,
                    "password": self.password,
                    "login": "Login",
                },
            )
        except RuntimeError as exc:
            print(f"✗ 登录前网络检查失败: {exc}")
            return False

        
        if "logout" in response.text.lower():
            print("✓ 登录成功！")
            return True
        if self._looks_like_login_page(response.text):
            print("✗ 登录失败，请检查账号密码")
            return False
        print("⚠ 登录状态无法明确判断，请先用调试工具确认")
        return False

    def get_file_list(self, district_id, data_type, year):
        """
        获取指定年份的所有文件列表
        返回: [{'name', 'id', 'size'}, ...]
        """
        api_url = (
            f"{self.base_url}?srq=clearinghouse&district_id={district_id}"
            f"&geotag=null&yy={year}&type={data_type}&returnformat=text"
        )
        files = []

        try:
            response = self._request("GET", api_url)
        except RuntimeError as exc:
            print(f"  获取 {year} 年文件列表失败: {exc}")
            return files

        if self._looks_like_login_page(response.text):
            print("  接口返回登录页，当前会话未登录或已失效")
            return files

        try:
            data = json.loads(response.text)
            for month, file_list in data.get("data", {}).items():
                _ = month
                for file_info in file_list:
                    files.append({
                        "name": file_info["file_name"],
                        "id": file_info["file_id"],
                        "size": file_info.get("bytes", 0),
                    })
        except json.JSONDecodeError:
            print(f"  解析 JSON 失败: {response.text[:200]}")
        return files

    def get_all_files(self, district_id, data_type, start_year=2018, end_year=2025):
        """获取多年的所有文件"""
        all_files = []
        print(f"正在获取 {start_year}-{end_year} 年的文件列表...")
        for year in range(start_year, end_year + 1):
            files = self.get_file_list(district_id, data_type, year)
            if files:
                print(f"  {year}年: {len(files)} 个文件")
                all_files.extend(files)

        seen = set()
        unique_files = []
        for file_info in all_files:
            if file_info["name"] not in seen:
                seen.add(file_info["name"])
                unique_files.append(file_info)

        unique_files.sort(key=lambda item: item["name"])
        print(f"\n总计: {len(unique_files)} 个文件")
        return unique_files

    def download_file(self, file_info, save_dir):
        """
        下载单个文件到本地临时目录
        """
        filename = file_info["name"]
        filepath = os.path.join(save_dir, filename)

        if os.path.exists(filepath):
            print(f"  跳过 (已存在): {filename}")
            return True

        download_url = f"{self.base_url}?download={file_info['id']}&dnode=Clearinghouse"
        try:
            response = self._request(
                "GET",
                download_url,
                timeout=(self.connect_timeout, self.download_timeout),
                stream=True,
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type.lower():
                preview = response.text[:2000]
                if self._looks_like_login_page(preview):
                    print(f"  ✗ {filename} - 需要重新登录")
                    return False

            total = int(response.headers.get("content-length", 0))
            with open(filepath, "wb") as file_obj:
                downloaded = 0
                for chunk in response.iter_content(8192):
                    if not chunk:
                        continue
                    file_obj.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r  下载中: {filename} ({pct}%)", end="", flush=True)

            size_kb = os.path.getsize(filepath) / 1024
            print(f"\r  ✓ {filename} ({size_kb:.1f} KB)" + " " * 20)
            return True

        except RuntimeError as exc:
            print(f"\n  ✗ 失败: {filename} - {exc}")
            return False
        except Exception as exc:
            print(f"\n  ✗ 失败: {filename} - {exc}")
            return False

    def batch_download(self, district_id, data_type, start_year=2018, end_year=2025):
        """
        按年份批量下载到本地目录
        参数:
            district_id:  区域ID (3,4,5,6,7,8,10,11,12)
            data_type:    数据类型 (meta / station_5min / station_hour / station_day)
            start_year / end_year: 年份范围
        """
        success = 0
        total = 0
        for year in range(start_year, end_year + 1):
            save_dir = build_data_dir(district_id, year, data_type)
            os.makedirs(save_dir, exist_ok=True)
            files = self.get_file_list(district_id, data_type, year)
            if not files:
                print(f"⚠ {year} 年未找到文件")
                continue

            print(f"\n开始下载到目录: {save_dir}\n")
            year_success = 0
            for index, file_info in enumerate(files, 1):
                print(f"[{year} {index}/{len(files)}]", end="")
                if self.download_file(file_info, str(save_dir)):
                    success += 1
                    year_success += 1
                total += 1
                time.sleep(1)

            print(f"✓ {year} 年下载完成: {year_success}/{len(files)} 个文件")

        print(f"\n✓ 全部下载完成: {success}/{total} 个文件")
        return success

    def download_single_year(self, district_id, data_type, year):
        """只下载单个年份的数据到本地目录"""
        save_dir = build_data_dir(district_id, year, data_type)
        os.makedirs(save_dir, exist_ok=True)
        files = self.get_file_list(district_id, data_type, year)
        if not files:
            print("⚠ 未找到文件")
            return 0

        print(f"\n开始下载到目录: {save_dir}\n")
        success = 0
        for index, file_info in enumerate(files, 1):
            print(f"[{index}/{len(files)}]", end="")
            if self.download_file(file_info, str(save_dir)):
                success += 1
            time.sleep(1)

        print(f"\n✓ 下载完成: {success}/{len(files)} 个文件")
        return success


downloader = PeMSDownloader(USERNAME, PASSWORD)
print("✓ 下载器已准备好")


if __name__ == "__main__":
    # 第四步：登录
    if not downloader.login():
        raise SystemExit(1)

    # 第五步：下载数据
    # 下载 District 4 元数据到 /data/yuzhang_fei/PEMS/PEMSD4_2025/d04_meta
    downloader.download_single_year(
        district_id=5,
        data_type="meta",
        year=2025,
    )

    # 如需下载 5 分钟数据，可取消下面注释
    # downloader.download_single_year(
    #     district_id=5,
    #     data_type="station_5min",
    #     year=2025,
    # )

    # 自定义下载示例
    # DISTRICT   = 4
    # DATA_TYPE  = 'meta'
    # START_YEAR = 2020
    # END_YEAR   = 2025
    #
    # downloader.batch_download(
    #     district_id=DISTRICT,
    #     data_type=DATA_TYPE,
    #     start_year=START_YEAR,
    #     end_year=END_YEAR,
    # )