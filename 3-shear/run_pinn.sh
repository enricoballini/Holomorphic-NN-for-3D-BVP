export JAX_TRACEBACK_FILTERING=off

python3 -u geometry_surface_mesh_pinn.py
python3 -u case_settings_pinn.py
python3 -u train_adam_pinn.py
python3 -u plot_losses_pinn.py
python3 -u postprocess_error_pinn.py
python3 -u plot_and_make_vtu_err_pinn.py
python3 -u plot_and_make_vtu_pinn.py

