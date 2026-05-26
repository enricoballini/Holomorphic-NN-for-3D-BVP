export JAX_TRACEBACK_FILTERING=off

rm -r "./data"
rm -r "./results"
mkdir "./data"
mkdir "./results"

python3 geometry_surface_mesh.py
python3 plot_and_make_vtu_geometry.py

python3 main.py
python3 plot_losses.py
python3 plot_mins_maxes.py

python3 convert_abaqus_to_numpy.py
python3 postprocess_error.py

python3 plot_and_make_vtu.py
python3 plot_and_make_vtu_err_hol.py
python3 plot_and_test_lapl.py

python3 plot_and_make_vtu_loss.py

python3 plot_and_make_vtu_fem.py