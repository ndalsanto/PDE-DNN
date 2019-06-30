import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, '..')

        
        

# class aimed at hosting the PDE activation in the network and anything needed to use it 

class navier_stokes_pde_activation:

    def __init__( self, _rb_manager, _number_of_output, _output_locations, \
                  _num_pde_activation_input, theta_min, theta_max, \
                  _num_pde_f_activation_input = 0, theta_f_min=np.array(0), theta_f_max=np.array(0), \
                  num_nonlinear_terms=0, nl_min=np.array(0), nl_max=np.array(0), \
                  _num_total_theta_as=0, _num_total_theta_fs=0 ):
        
        self.M_num_total_theta_as = _num_total_theta_as
        self.M_num_total_theta_fs = _num_total_theta_fs

        self.M_theta_min = theta_min
        self.M_theta_max = theta_max
        self.M_theta_f_min = theta_f_min
        self.M_theta_f_max = theta_f_max
        
        self.M_num_pde_activation_input   = _num_pde_activation_input
        self.M_num_pde_f_activation_input = _num_pde_f_activation_input
        
        self.M_num_nonlinear_terms = num_nonlinear_terms
        
        self.M_nl_min = nl_min
        self.M_nl_max = nl_max
        
        self.prepare_selectors( _rb_manager, _number_of_output, _output_locations )
        
        self.import_affine_arrays( _rb_manager )
        
        return

    def prepare_selectors( self, _rb_manager, _number_of_output, _output_locations ):
        
        print( "Preparing selectors for handling the RB problem" )

        self.M_A_mu_selectors = []

        # this is for the theta_a
        # for the viscosity
        # nonlinear terms
        # theta_f
        total_activation = self.M_num_pde_activation_input   \
                         + 1                                 \
                         + self.M_num_nonlinear_terms        \
                         + self.M_num_pde_f_activation_input 

        for iSelector in range( total_activation ):
            A_np_selector = np.zeros( ( total_activation, 1 ) )
            A_np_selector[iSelector, 0] = 1
            self.M_A_mu_selectors.append( tf.add( tf.zeros( A_np_selector.shape ), A_np_selector ) )

        print( "Total activation is %d " % total_activation )

        VT_np_output = np.transpose( _rb_manager.get_basis( _output_locations ) )
        self.M_N = VT_np_output.shape[0]
        self.M_VT_output = tf.convert_to_tensor( VT_np_output, dtype=tf.float32 )

        A_enlarge_mu_np = np.zeros( ( total_activation, _number_of_output ) )

        for iSelector in range( total_activation ):
            starting_point = -total_activation+iSelector
            A_enlarge_mu_np[iSelector, starting_point] = 1
        
        self.M_A_enlarge_mu = tf.add( tf.zeros( ( total_activation, _number_of_output ) ), A_enlarge_mu_np )

        A_enlarge_nl_np = np.zeros( ( self.M_num_nonlinear_terms, _number_of_output ) )

        for jSelector in range( self.M_num_nonlinear_terms ):
            starting_point = -self.M_num_nonlinear_terms + jSelector
            A_enlarge_nl_np[ jSelector, starting_point ] = 1
        
        self.M_A_enlarge_nonlinearities = tf.add( tf.zeros( ( self.M_num_nonlinear_terms, _number_of_output ) ), A_enlarge_nl_np )

        number_of_output_locations = _output_locations.shape[0]
        print( "number_of_output_locations is %d and total number of output is %d " % ( number_of_output_locations, _number_of_output ) )

        A_enlarge_f_np = np.zeros( ( number_of_output_locations, _number_of_output ) )
        A_enlarge_f_np[:, np.arange(number_of_output_locations)] = np.identity( number_of_output_locations )
        self.M_A_enlarge_f = tf.convert_to_tensor( A_enlarge_f_np, dtype=tf.float32 )

        A_select_all_parameters_np = np.zeros( ( total_activation, total_activation ) )

        for ii in range( self.M_num_pde_activation_input + 1 + self.M_num_pde_f_activation_input ):
            A_select_all_parameters_np[ii, ii] = 1.0

        A_select_all_nonlinearities_np = np.zeros( ( total_activation, total_activation ) )

        for ii in range( self.M_num_pde_activation_input + 1 + self.M_num_pde_f_activation_input, total_activation ):
            A_select_all_nonlinearities_np[ii, ii] = 1.0

        self.M_A_select_all_parameters     = tf.add( tf.zeros( ( total_activation, total_activation ) ), A_select_all_parameters_np )
        self.M_A_select_all_nonlinearities = tf.add( tf.zeros( ( total_activation, total_activation ) ), A_select_all_nonlinearities_np )

        return
        
    def  import_affine_arrays( self, _rb_manager ):
        
        self.M_rb_affine_matrices = []
        self.M_rb_affine_vectors = []

        self.M_Qa = _rb_manager.get_Qa( )
        self.M_Qf = _rb_manager.get_Qf( )

        # importing rb arrays and expanding tensors to handle the computations of RB linear systems at once
        for iQa in range( _rb_manager.get_Qa( ) ):
            self.M_rb_affine_matrices.append( tf.convert_to_tensor( _rb_manager.get_rb_affine_matrix( iQa ), dtype=tf.float32 ) )
            self.M_rb_affine_matrices[iQa] = tf.expand_dims(self.M_rb_affine_matrices[iQa], 0 )

        for iQf in range( _rb_manager.get_Qf( ) ):
            self.M_rb_affine_vectors.append( tf.convert_to_tensor( _rb_manager.get_rb_affine_vector( iQf ), dtype=tf.float32 ) )
            self.M_rb_affine_vectors[iQf] = tf.expand_dims(self.M_rb_affine_vectors[iQf], 0 )
            self.M_rb_affine_vectors[iQf] = tf.expand_dims(self.M_rb_affine_vectors[iQf], 2 )

        return
    
    M_num_pde_activation_input = 0
    M_num_pde_f_activation_input = 0
    
    M_N = 0
    M_min_param   = np.zeros( 0 )
    M_range_param = np.zeros( 0 )

    M_VT_output = tf.zeros( (0,0) )
    
    M_A_mu_selectors = []
    
    M_A_enlarge_mu = tf.zeros( (0,0) )
    M_A_enlarge_f = tf.zeros( (0,0) )

    M_rb_affine_matrices = []
    M_rb_affine_vectors = []

    M_theta_min = 0
    M_theta_max = 0
    M_theta_f_min = 0
    M_theta_f_max = 0
    M_num_nonlinear_terms = 0
    
    M_Qa = 0
    M_Qf = 0
    M_num_total_theta_as = 0
    M_num_total_theta_fs = 0
    
    M_A_enlarge_nonlinearities = None
    
    
    def pde_mu_solver( self, _computed_theta_mu ):

        theta_mu_shape = tf.shape( _computed_theta_mu ) 
        theta_mu_length = theta_mu_shape[0]

        ns = tf.ones( theta_mu_length ) 
        ns = tf.reshape( ns, (theta_mu_length, 1, 1) )

        # store parameters (AND NOT THE NONLINEARITIES) in a bigger matrix, ready to be summed 
        pde_mu_solver_tf_output_param = tf.matmul( _computed_theta_mu, self.M_A_enlarge_mu )
        
        rb_matrix_online = tf.zeros( self.M_rb_affine_matrices[0].shape )
        
        # affine components coming from mdeim
        for iQa in range( self.M_num_pde_activation_input ):
            rb_matrix_online = tf.add( rb_matrix_online,    \
                                       ( tf.expand_dims( self.M_theta_min[iQa] 
                                                       + (self.M_theta_max[iQa] - self.M_theta_min[iQa]) 
                                                       * tf.matmul( _computed_theta_mu, self.M_A_mu_selectors[iQa] ), 1 ) ) \
                                       * ( ns * self.M_rb_affine_matrices[iQa] ) )

        rb_matrix_online = tf.add( rb_matrix_online, ns * self.M_rb_affine_matrices[self.M_num_total_theta_as] )
        
        # the position of viscosity in _computed_theta_mu is just after the theta_a 
        # and its boundaries are the last ones in theta_min and theta_max
        rb_matrix_online = tf.add( rb_matrix_online,    \
                                   ( tf.expand_dims( self.M_theta_min[-1] 
                                                   + (self.M_theta_max[-1] - self.M_theta_min[-1]) 
                                                   * tf.matmul( _computed_theta_mu, self.M_A_mu_selectors[self.M_num_pde_activation_input] ), 1 ) ) \
                                   * ( ns * self.M_rb_affine_matrices[self.M_num_total_theta_as+1] ) * 0.01 )

        for iQn in range( self.M_num_nonlinear_terms ):
            rb_matrix_online = tf.add( rb_matrix_online,    \
                                       ( tf.expand_dims( self.M_nl_min[iQn] 
                                                   + (self.M_nl_max[iQn] - self.M_nl_min[iQn]) 
                                                   * tf.matmul( _computed_theta_mu, self.M_A_mu_selectors[self.M_num_pde_activation_input+1+self.M_num_pde_f_activation_input+iQn] ), 1 ) ) \
                                       * ( ns * self.M_rb_affine_matrices[self.M_num_total_theta_as + 2 + iQn] ) )

        rb_rhs_online = tf.zeros( self.M_rb_affine_vectors[0].shape )
        
        # affine components coming from deim
        for iQf in range( self.M_num_pde_f_activation_input ):
            rb_rhs_online = tf.add( rb_rhs_online,    \
                           ( tf.expand_dims( self.M_theta_f_min[iQf] 
                                           + (self.M_theta_f_max[iQf] - self.M_theta_f_min[iQf]) 
                                           * tf.matmul( _computed_theta_mu, self.M_A_mu_selectors[self.M_num_pde_activation_input+1+iQf] ), 1 ) ) \
                           * ( ns * self.M_rb_affine_vectors[iQf] ) )

        # adding viscoity of lifting 
        rb_rhs_online = tf.add( rb_rhs_online,    \
                       ( tf.expand_dims( self.M_theta_min[-1] 
                                       + (self.M_theta_max[-1] - self.M_theta_min[-1]) 
                                       * tf.matmul( _computed_theta_mu, self.M_A_mu_selectors[self.M_num_pde_activation_input] )*0.01, 1 ) ) \
                       * ( ns * self.M_rb_affine_vectors[self.M_num_total_theta_fs] ) )

        rb_rhs_online = tf.add( rb_rhs_online, ns * self.M_rb_affine_vectors[self.M_num_total_theta_fs+1] )

        rb_sol_online = tf.matrix_solve( rb_matrix_online, rb_rhs_online )

        # this provides the transpose of a matrix containing all the RB solutions, i.e. [u_n(mu_0), ..., u_n(mu_ns)]^T
        rb_sol_online_2 = tf.reshape( rb_sol_online, (theta_mu_length, self.M_N ) )

        pde_mu_solver_tf_output_0 = tf.matmul( rb_sol_online_2, self.M_VT_output ) #/ tf.matmul( ns_2d, self.M_scaling )

        pde_mu_solver_tf_output = tf.matmul( pde_mu_solver_tf_output_0, self.M_A_enlarge_f )

        pde_mu_solver_tf_output = tf.add( pde_mu_solver_tf_output, pde_mu_solver_tf_output_param )

        return pde_mu_solver_tf_output


