# 1. Install Python dependencies
conda create -n IPR python==3.9
conda activate IPR

# pip install vllm==0.4.0.post1
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

cd textworld
pip install .
cd ..

pip install -r requirements.txt
pip install "fschat[model_worker]"
pip install -e ".[train]"

cd eval_agent
pip install -r requirements.txt
cd ..

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

cd envs/webshop
pip install -e .
python -m spacy download en_core_web_lg
conda install -y -c conda-forge openjdk=11

# 2. Download data for WebShop environment
gdown https://drive.google.com/uc?id=1G_0ccLWn5kZE5rpeyAdh_YuoNzvBUjT9
gdown https://drive.google.com/uc?id=11zOUDkJSgGhYin9NxQtG8PVpDsika86y
unzip data.zip
mkdir search_index
unzip indexes.zip -d search_index/

# 3. Download data for ALFWorld environment
cd ../..
cd eval_agent/data/alfworld
gdown https://drive.google.com/uc?id=1y7Vqeo0_xm9d3I07vZaP6qbPFtyuJ6kI
unzip alfworld_data.zip

# 4. Download data for InterCodeSQL environment
cd ../../..
cd eval_agent/data/intercode_sql
gdown https://drive.google.com/uc?id=19AyZnrniD_NXSbV8mHPh5FgoXjN-5WvP
unzip intercodesql_data.zip

# 5. Create docker image for SQL environment
cd ../../..
echo "Setting up docker image for sql..."
docker-compose -f eval_agent/data/intercode_sql/sql-docker-compose.yml up -d

# 6. Download expert trajectories for supervised fine-tuning and mixture trajectory optimization
gdown https://drive.google.com/uc?id=1mkFVIbpeR-bgIzYnIkpYkIeRH6GlFGEs
unzip IPR_data.zip