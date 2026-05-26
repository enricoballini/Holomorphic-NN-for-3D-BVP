export JAX_TRACEBACK_FILTERING=off

mkdir "./data"

python3 -u geometry_surface_mesh.py
python3 -u case_settings.py
python3 -u plot_losses.py
python3 -u plot_mins_maxes.py

# python3 -u geometry_surface_mesh_pinn.py
# python3 -u case_settings_pinn.py
# python3 -u plot_losses_pinn.py

python3 -u postprocess_error.py
python3 -u test_equilibrium.py

python3 -u plot_and_make_vtu_bc.py
python3 -u plot_and_make_vtu_hol.py
python3 -u plot_and_make_vtu_exact.py
python3 -u plot_and_make_vtu_err_hol.py
# python3 -u plot_and_make_vtu_err_pinn.py

python3 -u plot_and_make_vtu_loss.py