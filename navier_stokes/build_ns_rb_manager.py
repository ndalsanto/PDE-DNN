#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:06:51 2019

@author: Niccolo' Dal Santo
@email : niccolo.dalsanto@epfl.ch
"""

import numpy as np
import pyorb_core.utils.array_utils as pyorb_array_utils
import pyorb_core.rb_library.m_deim as m_deim
import pyorb_core.rb_library.rb_manager as rm
import pyorb_core.rb_library.affine_decomposition as ad
import pyorb_core.pde_problem.parameter_generator as pg

def build_ns_rb_manager( my_ns, do_rb_offline, base_folder, my_parameter_handler, fem_additional_parameters ):

    mesh = fem_additional_parameters['mesh_name']
    offline_selection = fem_additional_parameters['offline_selection']
    m_deim_mu_grid = fem_additional_parameters['m_deim_mu_grid']
    ns_m_deim = fem_additional_parameters['ns_m_deim']
    n_s = fem_additional_parameters['n_s']
    ns_test = fem_additional_parameters['ns_test']
    rb_tol = fem_additional_parameters['rb_tol']
    pod_metric = fem_additional_parameters['pod_metric']
    
    my_mdeim = m_deim.Mdeim( my_ns )
    my_deim = m_deim.Deim( my_ns )
    
    theta_mdeim_min = np.zeros( 0 )
    theta_mdeim_max = np.zeros( 0 )
    theta_deim_min = np.zeros( 0 )
    theta_deim_max = np.zeros( 0 )
    
    if offline_selection == 'tensor':
        m_deim_parameter_generator = pg.Tensor_parameter_generator( 2, m_deim_mu_grid )
        my_parameter_handler.substitute_parameter_generator( m_deim_parameter_generator )
    
    if do_rb_offline == 1:
        my_mdeim.set_save_offline( True, base_folder + '/' )
        my_mdeim.perform_mdeim( ns_m_deim, 10**(-6) )
    
        theta_mdeim_min, theta_mdeim_max = my_mdeim.compute_theta_min_max( )
        pyorb_array_utils.save_vector( theta_mdeim_min, base_folder + 'theta_mdeim_min.txt' )
        pyorb_array_utils.save_vector( theta_mdeim_max, base_folder + 'theta_mdeim_max.txt' )
    
        pyorb_array_utils.save_matrix( my_mdeim.M_snapshots_matrix, base_folder + 'matrix_snapshots.txt' )
        pyorb_array_utils.save_matrix( my_mdeim.M_snapshots_coefficients, base_folder + 'matrix_snapshots_theta.txt' )
    
    else:
        my_mdeim.load_mdeim_offline( base_folder )
        theta_mdeim_min = np.loadtxt( base_folder + 'theta_mdeim_min.txt' )
        theta_mdeim_max = np.loadtxt( base_folder + 'theta_mdeim_max.txt' )
        my_mdeim.M_snapshots_coefficients = np.loadtxt( base_folder + 'matrix_snapshots_theta.txt' )
        my_mdeim.M_snapshots_matrix = np.loadtxt( base_folder + 'matrix_snapshots.txt' )
    
    my_mdeim.M_snapshots_coefficients
    num_mdeim_affine_components_A = my_mdeim.get_num_mdeim_basis( )
    
    if do_rb_offline == 1:
        my_deim.set_save_offline( True, base_folder + '/' )
        my_deim.perform_deim( ns_m_deim, 10**(-6) )
        theta_deim_min, theta_deim_max = my_deim.compute_theta_min_max( )
        pyorb_array_utils.save_vector( theta_deim_min, base_folder + 'theta_deim_min.txt' )
        pyorb_array_utils.save_vector( theta_deim_max, base_folder + 'theta_deim_max.txt' )
    
        pyorb_array_utils.save_matrix( my_deim.M_snapshots_matrix, base_folder + 'vector_snapshots.txt' )
        pyorb_array_utils.save_matrix( my_deim.M_snapshots_coefficients, base_folder + 'vector_snapshots_theta.txt' )
    
    else:
        my_deim.load_deim_offline( base_folder + '/' )
        theta_deim_min = np.loadtxt( base_folder + 'theta_deim_min.txt' )
        theta_deim_max = np.loadtxt( base_folder + 'theta_deim_max.txt' )
        my_deim.M_snapshots_coefficients = np.loadtxt( base_folder + 'vector_snapshots_theta.txt' )
        my_deim.M_snapshots_matrix = np.loadtxt( base_folder + 'vector_snapshots.txt' )
        
    my_deim.M_snapshots_coefficients
    num_deim_affine_components_f = my_deim.get_num_basis( )
        
    my_ns.set_mdeim( my_mdeim )
    my_ns.set_deim( my_deim )
    
    print( rm.__doc__ )
    my_rb_manager = rm.RbManager( my_ns )
    
    SAVE_OFFLINE = 1
    
    name = ''
    
    if SAVE_OFFLINE == 1:
        my_rb_manager.save_offline_structures( base_folder + name + "snapshots_" + mesh + '.txt', \
                                               base_folder + name + "basis_" + mesh + '.txt', \
                                               base_folder + name + "rb_affine_components_" + mesh, \
                                               base_folder + name + 'offline_parameters.data' )
    
    if do_rb_offline == 1:
        my_rb_manager.build_snapshots( n_s, seed=456 )
        
        my_rb_manager_test = rm.RbManager( my_ns )
        
        if SAVE_OFFLINE == 1:
            my_rb_manager_test.save_offline_structures( base_folder + 'test_snapshots_' + mesh + '.txt', \
                                                        base_folder + 'basis_test_' + mesh + '.txt', \
                                                        base_folder + 'rb_affine_components_test_' + mesh, \
                                                        base_folder + 'test_offline_parameters.data' )
        
        my_rb_manager_test.build_snapshots( ns_test, seed=1234 )
    
    else:
        my_rb_manager.import_snapshots_matrix( base_folder + name + "snapshots_" + mesh + '.txt' )
        my_rb_manager.import_snapshots_parameters( base_folder + 'offline_parameters.data' )
    
    n_s = my_rb_manager.M_snapshots_matrix.shape[1]
    
    my_rb_manager.import_test_parameters( base_folder + 'test_offline_parameters.data' )
    my_rb_manager.import_test_snapshots_matrix( base_folder + 'test_snapshots_' + mesh + '.txt' )
    
    if do_rb_offline == 1:
        my_rb_manager.perform_pod( rb_tol, pod_metric )
    else:
        my_rb_manager.import_basis_matrix( base_folder + name + "basis_" + mesh + '.txt' )
    
    # want_to_solve_newton should be 1 to consider the affine components for the Jacobian
    want_to_solve_newton = 0
    want_to_solve_newton = 1
    
    # defining the affine decomposition structure
    # the two + 1 in the affine decoimpositions are the diffusion term and the lifting contribution in the nonlinear term
    my_affine_decomposition = ad.AffineDecompositionHandler( )
    my_affine_decomposition.set_Q( num_mdeim_affine_components_A + (1+want_to_solve_newton)*my_rb_manager.get_number_of_basis()+1+1, \
                                   num_deim_affine_components_f+1+1 )               # number of affine terms
    
    # we externally set the affine components for A, the ones for f are handled in the solver
    my_affine_decomposition.set_affine_a( my_mdeim.get_basis_list( ) )
    my_affine_decomposition.set_affine_f( my_deim.get_deim_basis_list( ) )
    
    my_rb_manager.set_affine_decomposition_handler( my_affine_decomposition )
    
    if do_rb_offline == 1:
        rb_functions_dict = my_rb_manager.get_rb_functions_dict( )
        my_ns.update_fom_specifics( rb_functions_dict )
    
        my_rb_manager.build_rb_affine_decompositions( _build_rb_tpl=True )
        
        if SAVE_OFFLINE == 1:
            my_rb_manager.save_rb_affine_decomposition( )
    
        my_ns.clear_fom_specifics( rb_functions_dict )
    
    else:
        my_affine_decomposition.import_rb_affine_matrices( base_folder + 'rb_affine_components_' + mesh + '_A' )
        my_affine_decomposition.import_rb_affine_vectors(  base_folder + 'rb_affine_components_' + mesh + '_f' )

    m_deim_min_max_collector = { \
                                'theta_mdeim_min' : theta_mdeim_min, \
                                'theta_mdeim_max' : theta_mdeim_max, \
                                'theta_deim_min'  : theta_deim_min, \
                                'theta_deim_max'  : theta_deim_max \
                                }

    return my_rb_manager, m_deim_min_max_collector
