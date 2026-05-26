export JAX_TRACEBACK_FILTERING=off

mkdir ./data

python3 -u case_settings_pinn.py

python3 -u plot_losses_pinn.py
python3 -u plot_mins_maxes_pinn.py

python3 -u convert_abaqus_to_numpy.py

python3 -u postprocess_error_pinn.py
python3 -u plot_and_make_vtu_err_pinn.py
python3 -u plot_and_make_vtu_pinn.py

