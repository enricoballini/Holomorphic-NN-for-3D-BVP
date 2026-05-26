export JAX_TRACEBACK_FILTERING=off

python3 plot_losses.py
python3 plot_mins_maxes.py

python3 convert_abaqus_to_numpy.py
python3 postprocess_error_pinn.py

python3 plot_and_make_vtu.py
python3 plot_and_make_vtu_err_pinn.py
python3 plot_and_test_lapl.py
