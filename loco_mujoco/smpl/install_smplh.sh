#!/bin/bash

# Function to read LOCO_MUJOCO_PATH from Python package
function get_loco_mujoco_path() {
    LOCO_MUJOCO_PATH=$(python -c "import loco_mujoco; print(loco_mujoco.__path__[0])" 2>/dev/null | tail -n 1)
    if [ -z "$LOCO_MUJOCO_PATH" ]; then
        echo "Error: Could not determine LOCO_MUJOCO_PATH from the installed loco_mujoco package."
        exit 1
    fi
    echo "$LOCO_MUJOCO_PATH"
}

# Set LOCO_MUJOCO_PATH dynamically
LOCO_MUJOCO_PATH=$(get_loco_mujoco_path)

# Define paths based on LOCO_MUJOCO_PATH
GENERATE_SCRIPT_PATH="$LOCO_MUJOCO_PATH/smpl/generate_smplh_model.py"
SMPL_CONF_PATH="$LOCO_MUJOCO_PATH/LOCOMUJOCO_VARIABLES.yaml"

# Check if the SMPL configuration file exists
if [ ! -f "$SMPL_CONF_PATH" ]; then
    echo "Error: Could not load the SMPL configuration file."
    echo "Please use the command:"
    echo "  loco-mujoco-set-smpl-model-path --path \"/path/to/smpl/models\""
    exit 1
fi

# Check if the Conda environment already exists
if conda info --envs | grep -q "^smplh_model_conversion_env\s"; then
    echo "Error: Conda environment 'smplh_model_conversion_env' already exists."
    echo "Please remove or rename it."
    exit 1
fi

# Run commands
{
    # Create Conda environment
    conda create --name smplh_model_conversion_env python=3.10 -y

    # Install required Python packages
    conda run --name smplh_model_conversion_env pip install 'numpy<1.23.0' chumpy tqdm pyyaml

    # Run the Python script
    conda run --name smplh_model_conversion_env python "$GENERATE_SCRIPT_PATH" --smpl-conf-file "$SMPL_CONF_PATH"

    # Clean up: remove the Conda environment
    conda remove --name smplh_model_conversion_env --all -y

    echo "Generation complete!"

} || {
    # Handle errors
    echo -e "\nAn error occurred during the SMPL-H model generation process."
    echo "Please check the errors above and convert the SMPL-H model manually."
    exit 1
}
