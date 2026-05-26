Setup
-----

Activate the Python environment and install the required dependencies:

pip install -r requirements.txt


Running Test Cases
------------------

Run all test cases:

./run_all_cases.sh

Run a single test case:

cd [case_name]
./run.sh


Output
------

Generated files are written to:

- [case_name]/data/
- [case_name]/results/


Notes
-----

Each test case is completely independent. Cases can be modified, copied,
removed, or executed individually without affecting the others.