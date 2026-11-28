"""Set up the tests: one BLAS thread and ROS domain 93 (never 0, the real robots' bus)."""
import os

for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_var, '1')
os.environ['ROS_DOMAIN_ID'] = '93'
