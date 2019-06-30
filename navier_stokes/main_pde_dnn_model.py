#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:05:30 2019

@author: Niccolo' Dal Santo
@email : niccolo.dalsanto@epfl.ch
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import random
import tensorflow as tf
print(tf.__version__)

tf.enable_eager_execution( )

import time 

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../../pyorb/')
sys.path.insert(0, '../../pyorb/examples/navier_stokes')

print(sys.path)

print( "This is the name of the script: %s " % sys.argv[0] )
print( "Number of arguments: %d " % len(sys.argv) )
print( "The arguments are: %s " % str(sys.argv) )

import pyorb_core.tpl_managers.external_engine_manager as mee

if len(sys.argv) > 1:
    USED_THETAS_A = int( sys.argv[1] )
    number_of_fem_coordinates = int( sys.argv[2] )
else:
    USED_THETAS_A = 2
    number_of_fem_coordinates = 100
    
USED_THETAS_F = USED_THETAS_A
USED_NONLINEAR_TERMS = USED_THETAS_A

print( '%d %d ' % (USED_THETAS_A, number_of_fem_coordinates) )

matlab_library_path = '/usr/scratch/dalsanto/EPFL/DeepLearning/feamat/'
matlab_pyorb_interface = '/usr/scratch/dalsanto/EPFL/DeepLearning/pyorb-matlab-api/'

my_matlab_engine_manager = mee.external_engine_manager( 'matlab', matlab_library_path, matlab_pyorb_interface )
my_matlab_engine_manager.start_engine( )
my_matlab_external_engine = my_matlab_engine_manager.get_external_engine( )

import pyorb_core.pde_problem.parameter_handler as ph

mu0_min = 1.; mu0_max = 10. # defines a diffusion in the ns solver between 0.01 and 0.1
mu1_min = 0.; mu1_max = 0.1

param_range = ''

ns_test = 100
mesh = 'fine'
mesh = 'very_coarse'

do_rb_offline = 0
pod_metric = ''
offline_selection = 'tensor'

if mu1_max <= 0.250001:

    param_range = 'param_025_'
    ns_m_deim = 750
    n_s = 750
    mu_grid = np.array( [37, 51] )
    ns_test = 20
    m_deim_mu_grid = np.array( [1, 1000] )
    ns_m_deim = m_deim_mu_grid[0] * m_deim_mu_grid[1]
    rb_tol = 10**(-4)
    do_rb_offline = 0
    pod_metric = ''

# for using example with smaller parameter space
if mu1_max <= 0.101:
    param_range = 'param_010_'
    mu_grid = np.array( [10, 6] )
    offline_selection = 'tensor'
    n_s = mu_grid[0] * mu_grid[1]
    ns_m_deim = n_s
    ns_test = 10
    rb_tol = 10**(-4)
    do_offline = 0
    pod_metric = ''

param_min = np.array([mu0_min, mu1_min])
param_max = np.array([mu0_max, mu1_max])
num_parameters = param_min.shape[0]

import pyorb_core.pde_problem.parameter_generator as pg

if offline_selection == 'tensor':
    my_parameter_generator = pg.Tensor_parameter_generator( 2, mu_grid )
else:
    my_parameter_generator = pg.Random_parameter_generator( 2 )

# preparing the parameter handler
my_parameter_handler = ph.Parameter_handler( my_parameter_generator )
my_parameter_handler.assign_parameters_bounds( param_min, param_max )

# define the fem problem
import navier_stokes_problem as ns

fom_specifics = {
        'model': 'navier_stokes', \
        'simulation_name'   : 'navier_stokes_' + mesh,\
        'use_nonhomogeneous_dirichlet' : 'Y', \
        'mesh_name' : '/usr/scratch/dalsanto/EPFL/DeepLearning/pyorb_development/examples/navier_stokes/meshes/bifurcation_' + mesh + '.msh', \
        'full_path'         : '/usr/scratch/dalsanto/EPFL/DeepLearning/DLPDEs/elliptic_example/navier_stokes_2d/' }

my_ns = ns.navier_stokes_problem( my_parameter_handler, my_matlab_external_engine, fom_specifics )

base_folder = 'offline_' + param_range + offline_selection + '_' + mesh + '/'

#%%

fem_additional_parameters = { \
        'mesh_name' : mesh, \
        'offline_selection' : offline_selection, \
        'm_deim_mu_grid' : m_deim_mu_grid, \
        'ns_m_deim' : ns_m_deim, \
        'n_s' : n_s, \
        'ns_test' : ns_test, \
        'rb_tol' : rb_tol, \
        'pod_metric' : pod_metric }

import build_ns_rb_manager as bnrbm

my_rb_manager, m_deim_min_max_collector = bnrbm.build_ns_rb_manager( my_ns, do_rb_offline, base_folder, my_parameter_handler, fem_additional_parameters )

my_rb_manager.print_rb_offline_summary( )

test_parameters = my_rb_manager.M_test_parameters

my_rb_manager_for_samples = my_rb_manager

theta_mdeim_min = m_deim_min_max_collector['theta_mdeim_min']
theta_mdeim_max = m_deim_min_max_collector['theta_mdeim_max']
theta_deim_min = m_deim_min_max_collector['theta_deim_min']
theta_deim_max = m_deim_min_max_collector['theta_deim_max']
num_mdeim_affine_components_A = len( theta_mdeim_min)
num_deim_affine_components_f  = len( theta_deim_min )

#%%

import fem_data_generation as gd
#
#########################################################################
################### TRAINING AND TESTING DATA SETTING ###################
#########################################################################

# locations of points where the exact solution is obtained and employed as data
# mu_{in} = xy_locations
fem_dim = 0

# mesh sizes used in the numerical examples
if mesh == 'very_coarse':
    fem_dim = 8204
elif mesh == 'fine':
    fem_dim = 29843

number_of_non_dirichlet_points = fem_dim*2 - 1

sampling = 'random'

fem_coordinates = gd.generate_fem_coordinates( number_of_fem_coordinates, 0, number_of_non_dirichlet_points )

number_of_output_coordinates = 1600
fem_output_coordinates = gd.generate_fem_coordinates( number_of_output_coordinates, 0, number_of_non_dirichlet_points, \
                                                      possible_coordinates=[] )

number_of_output_coordinates = len(fem_output_coordinates)

# number of selected parameters for the training
my_rb_manager_for_samples.M_get_test = False
n_sample = my_rb_manager_for_samples.M_snapshots_matrix.shape[1]
X_train, y_train = gd.generate_fem_training_data( n_sample, fem_coordinates, fem_output_coordinates, my_rb_manager_for_samples )

# theta functions for stiffness + theta functions for rhs + viscosity + prediction for nonlinear velocity term
rb_solver_input_size = USED_THETAS_A + USED_THETAS_F + 1 + USED_NONLINEAR_TERMS

# enlarging the output in order to access the predicted theta functions and setting them to zero 
# (they are not penalized in the training anyway)
new_y_train = np.zeros( (y_train.shape[0], y_train.shape[1] + rb_solver_input_size) )
new_y_train[:, 0:len(fem_output_coordinates)] = y_train
y_train = new_y_train

# we use the same snapshots container that is the same rb_manager, however we switch its option to give 
# test snapsthos instead of train
my_rb_manager_for_samples.M_get_test = True
noise_magnitude = 0.0
X_test, y_test = gd.generate_fem_training_data( ns_test, fem_coordinates, fem_output_coordinates, \
                                                my_rb_manager_for_samples )

new_y_test = np.zeros( (y_test.shape[0], y_test.shape[1] + rb_solver_input_size) )
new_y_test[:, 0:len(fem_output_coordinates)] = y_test
y_test = new_y_test


#%%

##my_rb_manager_for_samples = my_rb_manager
#
#import generate_data as gd
#
#########################################################################
##################### TRAINING AND TEST DATA SETTING ####################
#########################################################################
#
## locations of points where the exact solution is obtained and employed as data
## mu_{in} = xy_locations
#fem_dim = 0
#
#if mesh == 'coarse':
#    fem_dim = 7592
#elif mesh == 'very_coarse':
#    fem_dim = 8204
#elif mesh == 'fine':
#    fem_dim = 29843
#    
#number_of_non_dirichlet_points = fem_dim*2 - 1
#
##sampling = 'tensorial'
#sampling = 'random'
#
#_possible_coordinates = []
#
#if mesh == 'fine':
#    _possible_coordinates = np.loadtxt( base_folder + mesh + '_projection.txt' ).astype(int)
#    my_possible_coordinates = np.zeros( 2*len(_possible_coordinates) )
#    my_possible_coordinates[0:len(_possible_coordinates)] = _possible_coordinates
#    my_possible_coordinates[len(_possible_coordinates):len(_possible_coordinates)*2] = _possible_coordinates + fem_dim
#    my_possible_coordinates = my_possible_coordinates.astype(int)
#
#my_possible_coordinates = []
#
#fem_coordinates = gd.generate_fem_coordinates( number_of_fem_coordinates, 0, number_of_non_dirichlet_points, \
#                                               possible_coordinates=my_possible_coordinates )
#fem_coordinates = np.sort( fem_coordinates )
#fem_coordinates = np.unique( fem_coordinates )
#
#number_of_output_coordinates = 1600
#fem_output_coordinates = gd.generate_fem_coordinates( number_of_output_coordinates, 0, number_of_non_dirichlet_points, \
#                                                      possible_coordinates=[] )
#
##    number_of_output_coordinates = number_of_output_coordinates[0] * number_of_output_coordinates[1]
#
##fem_output_coordinates = np.arange(0, 2*fem_dim)
#
#fem_output_coordinates = np.unique( fem_output_coordinates )
#fem_output_coordinates = np.sort( fem_output_coordinates )
#
##fem_output_coordinates = fem_coordinates
##number_of_output_coordinates = len( fem_output_coordinates )
#
#number_of_output_coordinates = len(fem_output_coordinates)
#
#USED_THETAS_SET_GENERATIONS = 0 
#
#theta_mins = np.zeros( (USED_THETAS_SET_GENERATIONS) )
#theta_maxs = np.zeros( (USED_THETAS_SET_GENERATIONS) )
#
##theta_mins[:USED_THETAS_A] = theta_min_train[:USED_THETAS_A]
##theta_maxs[:USED_THETAS_A] = theta_max_train[:USED_THETAS_A]
##
##theta_mins[-USED_THETAS_F:] = theta_f_min_train[:USED_THETAS_F]
##theta_maxs[-USED_THETAS_F:] = theta_f_max_train[:USED_THETAS_F]
#
## number of selected parameters for the training
#my_rb_manager_for_samples.M_get_test = False
#noise_magnitude = 0.0
#n_sample = my_rb_manager_for_samples.M_snapshots_matrix.shape[1]
##n_sample = 6000
#X_train, y_train = gd.generate_fem_training_data( n_sample, fem_coordinates, fem_output_coordinates, 
#                                                  my_rb_manager_for_samples, USED_THETAS_SET_GENERATIONS, \
#                                                  theta_mins, theta_maxs, \
#                                                  my_parameter_handler, noise_magnitude, \
#                                                  data_file=None, printParam=False )
#
## theta functions for stiffness + theta functions for rhs + viscosity + prediction for nonlinear velocity term
#rb_solver_input_size = USED_THETAS_A + USED_THETAS_F + 1 + USED_NONLINEAR_TERMS
#new_y_train = np.zeros( (y_train.shape[0], y_train.shape[1] + rb_solver_input_size) )
#new_y_train[:, 0:len(fem_output_coordinates)] = y_train
#y_train = new_y_train
#
#X_train_min = np.min(X_train[0:len(fem_output_coordinates)])
#X_train_max = np.max(X_train[0:len(fem_output_coordinates)])
#
#y_train_min = np.min(y_train[0:len(fem_output_coordinates)])
#y_train_max = np.max(y_train[0:len(fem_output_coordinates)])
#
##rescaled_y_train = y_train
##rescaled_y_train[0:number_of_fem_coordinates] = ( y_train - y_train_min ) / ( y_train_max - y_train_min )
#
##y_train = y_train[0:ns, :]
##X_train = X_train[0:ns, :]
##
###setting to zero the theta functions to avoid any learning from them
##y_train[:, -USED_THETAS:] = 0.0 * y_train[:, -USED_THETAS:]
##y_train[:, -USED_THETAS:-USED_THETAS_F] = rescaled_theta_train[0:ns, 0:USED_THETAS_A]
##y_train[:, -USED_THETAS_F:] = rescaled_theta_f_train[0:ns, 0:USED_THETAS_F]
##
###n_final_samples = 10000
###(X_out, y_out) = gd.expand_train_with_rb( X_train, y_train, n_final_samples, \
###                          fem_coordinates, fem_output_coordinates, \
###                          my_rb_manager, USED_THETAS, \
###                          param_min, param_max, my_ndp )
##
##
##theta_mins_test = np.zeros( (USED_THETAS) )
##theta_maxs_test = np.zeros( (USED_THETAS) )
##
##theta_mins_test[:USED_THETAS_A] = theta_min_test[:USED_THETAS_A]
##theta_maxs_test[:USED_THETAS_A] = theta_max_test[:USED_THETAS_A]
##
##theta_mins_test[-USED_THETAS_F:] = theta_f_min_test[:USED_THETAS_F]
##theta_maxs_test[-USED_THETAS_F:] = theta_f_max_test[:USED_THETAS_F]
#
#
#my_rb_manager_for_samples.M_get_test = True
#
#noise_magnitude = 0.0
#X_test, y_test = gd.generate_fem_training_data( ns_test, fem_coordinates, fem_output_coordinates, \
#                                                my_rb_manager_for_samples, USED_THETAS_SET_GENERATIONS,\
#                                                theta_mins, theta_maxs, \
#                                                my_parameter_handler, noise_magnitude, \
#                                                data_file=None, printParam=False )
#
#new_y_test = np.zeros( (y_test.shape[0], y_test.shape[1] + rb_solver_input_size) )
#new_y_test[:, 0:len(fem_output_coordinates)] = y_test
#y_test = new_y_test
#
#X_test_min = np.min(X_test[0:len(fem_output_coordinates)])
#X_test_max = np.max(X_test[0:len(fem_output_coordinates)])
#
#y_test_min = np.min(y_test[0:len(fem_output_coordinates)])
#y_test_max = np.max(y_test[0:len(fem_output_coordinates)])
#
##y_test[:, -USED_THETAS_SET_GENERATIONS:] = 0.0 * y_test[:, -USED_THETAS:]
##y_test[:, -USED_THETAS:-USED_THETAS_F] = rescaled_theta_test[0:ns, 0:USED_THETAS_A]
##y_test[:, -USED_THETAS_F:] = rescaled_theta_f_test[0:ns, 0:USED_THETAS_F]

#%%

# the theta values and the nonlinear terms are predicted in the range [0,1] (throiugh sigmoids)
# here we compute the bounds in order to rescale the prediction to the true interval in the 
# RB activation

used_theta_mdeim_min = np.zeros( USED_THETAS_A + 1 )
used_theta_mdeim_max = np.zeros( USED_THETAS_A + 1 )
used_theta_mdeim_min[0:USED_THETAS_A] = theta_mdeim_min[0:USED_THETAS_A]
used_theta_mdeim_max[0:USED_THETAS_A] = theta_mdeim_max[0:USED_THETAS_A]
used_theta_mdeim_min[-1] = mu0_min
used_theta_mdeim_max[-1] = mu0_max

used_theta_deim_min = np.zeros( USED_THETAS_F )
used_theta_deim_max = np.zeros( USED_THETAS_F )
used_theta_deim_min[0:USED_THETAS_F] = theta_deim_min[0:USED_THETAS_F]
used_theta_deim_max[0:USED_THETAS_F] = theta_deim_max[0:USED_THETAS_F]

VTu1 = np.dot( my_rb_manager.M_basis.T, my_rb_manager.get_snapshot(0) )
VTu2 = np.dot( my_rb_manager.M_basis.T, my_rb_manager.get_snapshot(1) )
VTu = np.dot ( my_rb_manager.M_basis.T, my_rb_manager.get_snapshots_matrix() )

u_min = np.zeros( VTu.shape[0] )
u_max = np.zeros( VTu.shape[0] )

for ii in range( len(u_max) ):
    u_max[ii] = np.max( VTu[ii, :] )
    u_min[ii] = np.min( VTu[ii, :] )

#%%

import pde_dnn_model as pdm

#%%

import navier_stokes_pde_activation as ns_pa

print( "######################################################################## ")
print( "############### TENSORFLOW + PDEs mu_{in} -> mu_{pde} ################## ")
print( "######################################################################## ")

EPOCHS = 200

my_pde_activation = ns_pa.navier_stokes_pde_activation( my_rb_manager, number_of_output_coordinates + rb_solver_input_size, \
                                                        fem_output_coordinates, \
                                                        USED_THETAS_A, used_theta_mdeim_min, used_theta_mdeim_max, \
                                                        _num_pde_f_activation_input=USED_THETAS_F, \
                                                        theta_f_min=used_theta_deim_min, theta_f_max=used_theta_deim_max, \
                                                        num_nonlinear_terms=USED_NONLINEAR_TERMS, nl_min=u_min, nl_max=u_max, \
                                                        _num_total_theta_as=num_mdeim_affine_components_A, \
                                                        _num_total_theta_fs=num_deim_affine_components_f )

#import the pde dnn model
import pde_dnn_model as pdm

# we take as input the input_fem_coordinates and 
# as output the output_fem_coordinates + the theta functions for the linear and nonlinear terms
my_pde_model = pdm.pde_dnn_model( number_of_fem_coordinates, number_of_output_coordinates + rb_solver_input_size )
my_pde_model.set_learning_rate( 0.001 )
my_pde_model.set_use_rb_activation( True )
my_pde_model.set_number_of_thetas( rb_solver_input_size )
my_pde_model.set_number_of_nonlinearities( USED_NONLINEAR_TERMS )
my_pde_model.set_rb_activation( my_pde_activation )

pde_model_name = 'pde_model_std_size.h5'
restore_pde_model = True

my_pde_model.build_model( network_width='std_size', restore_model=restore_pde_model, model_name=pde_model_name )

if restore_pde_model == False:
    my_pde_model.train_model( X_train, y_train, EPOCHS )
    my_pde_model.save_pde_dnn_model( pde_model_name )

my_pde_model.evaluate_model_errors( X_test, y_test )

#%%

start = time.time()
test_predictions = my_pde_model.predict( X_test )
end = time.time()
time_to_solve = end - start
print( 'Time to each predict RB solution with PDE-DNN %f ' % ( time_to_solve / float(ns_test) ) )

predicted_affine_decompositions = test_predictions[:, -rb_solver_input_size:]

#%%

errors = np.zeros( ns_test )
l2_errors = np.zeros( ns_test )

relative_errors = np.zeros( ns_test )
l2_relative_errors = np.zeros( ns_test )

n_ns_test = ns_test # 0 # ns_test

for ii in range( n_ns_test ):
    param = test_parameters[ii, :]
    print( '\nConsidering test parameter %d ' % (ii) )
    print( param )
    
    theta_a = np.zeros( num_mdeim_affine_components_A + 2 )
    theta_f = np.zeros( num_deim_affine_components_f  + 2 )

    # retrieve predictions \in [0, 1] of affine components and bring them to the original interval    
    for iQa in range( USED_THETAS_A ):
        theta_a[iQa]  = used_theta_mdeim_min[iQa] + ( used_theta_mdeim_max[iQa] - used_theta_mdeim_min[iQa] ) \
                                                    * predicted_affine_decompositions[ii, iQa]

    theta_a[-2] = 1.
    theta_a[-1] = 0.01 * ( used_theta_mdeim_min[-1] + ( used_theta_mdeim_max[-1] - used_theta_mdeim_min[-1] ) \
                                                      * predicted_affine_decompositions[ii, USED_THETAS_A] )

    for iQf in range( USED_THETAS_F ):
        theta_f[iQf]  = used_theta_deim_min[iQf] + ( used_theta_deim_max[iQf] - used_theta_deim_min[iQf] ) \
                                                   * predicted_affine_decompositions[ii, USED_THETAS_A+1+iQf]

    theta_f[-2] = 0.01 * ( used_theta_mdeim_min[-1] + ( used_theta_mdeim_max[-1] - used_theta_mdeim_min[-1] ) 
                                                      * predicted_affine_decompositions[ii, USED_THETAS_A] )
    theta_f[-1] = 1.
    
    uN_affine = np.zeros( USED_NONLINEAR_TERMS )
    
    for iQn in range( USED_NONLINEAR_TERMS ):
        uN_affine[iQn] = u_min[iQn] + ( u_max[iQn] - u_min[iQn] ) * predicted_affine_decompositions[ii, USED_THETAS_A+1+USED_THETAS_F+iQn]
    
    # compute RB solution using the prediction given by the MLP
    AN = my_ns.build_ns_rb_matrix( uN_affine, my_rb_manager.M_affineDecomposition, theta_a )
    fN = my_ns.build_ns_rb_vector( theta_f, my_rb_manager.M_affineDecomposition )
    uN = np.linalg.solve( AN, fN )

    # compute error wrt to the true FE solution
    s1 = my_rb_manager.get_test_snapshot( ii )
    utildeh = my_rb_manager.reconstruct_fem_solution( uN )
    e_h = s1 - utildeh
    
    l2_errors[ii] = np.linalg.norm(e_h)
    l2_relative_errors[ii] = l2_errors[ii] / np.linalg.norm(s1)
    print( 'Norm of relative error test %d is %f' % (ii, l2_relative_errors[ii] ) )

print( np.mean( l2_relative_errors[0:n_ns_test] ) )
print( np.max( l2_relative_errors[0:n_ns_test] ) )
print( l2_relative_errors[0:n_ns_test] )
