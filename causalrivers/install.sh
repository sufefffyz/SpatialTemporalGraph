conda env create -f causal_rivers_core.yml
wget https://github.com/CausalRivers/benchmark/releases/download/First_release/product.zip
unzip product
rm product.zip
echo "conda environment is installed as causalrivers. Done."
