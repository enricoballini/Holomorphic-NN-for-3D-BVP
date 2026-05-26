export JAX_TRACEBACK_FILTERING=off

rm -r "./data"
rm -r "./results"
mkdir "./data"
mkdir "./results"


python3 geometry_surface_mesh.py
python3 plot_and_make_vtu_geometry.py

python3 case_settings.py
python3 train.py
python3 plot_losses.py

python3 plot_and_make_vtu_bc.py
python3 plot_and_make_vtu_training_test_points.py

python3 postprocess_error.py
python3 plot_and_make_vtu_exact.py
python3 plot_and_make_vtu_hol.py
python3 plot_and_make_vtu_err_hol.py



