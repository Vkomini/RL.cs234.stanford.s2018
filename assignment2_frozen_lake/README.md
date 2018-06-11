# cs234n stanford RL HW2 Frozen Lake 
## Virtual env
cd cs234_hw2_FL
sudo pip install virtualenv      # This may already be installed
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
 # install gym
git clone https://github.com/openai/gym
cd gym
pip install -e . # minimal install
# Work on the assignment for a while ...
deactivate 