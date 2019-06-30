#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:18:27 2019

@author: Niccolo' Dal Santo
@email : niccolo.dalsanto@epfl.ch
"""

# tensorflow wrappers for PDE-DNN

import tensorflow as tf
from tensorflow import keras
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

# plot the history given by tf training
def plot_history(history, save_history=0, folder='./', name='default_history_plots_name'):

    import matplotlib as mlp
    font = {'family': 'sans-sertif', 'size' : 15 }
    mlp.rc('font', **font)
    
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.plot(history.epoch, np.array(history.history['loss']), label='train')
    plt.plot(history.epoch, np.array(history.history['val_loss']), label = 'validation')
    plt.legend( )
    plt.ylim((5*10**(-7), 5*10**(-2)))
    plt.tight_layout()

    if save_history==1:
        plt.savefig( folder + name + '.eps' )
  
    plt.show( )
    plt.close('all')


# custom loss which allows to access the theta function values given by the encoding part of the network (the mlp) but does not
# penalize them in the training
def custom_loss( weight_mu, number_of_thetas, number_of_nonlinearities, predicted_y, desired_y ):
    return tf.reduce_mean( tf.square( desired_y[:, :-number_of_thetas] - predicted_y[:, :-number_of_thetas] )  ) \
     + tf.reduce_mean( weight_mu  * tf.square(predicted_y[:, -number_of_thetas:-number_of_nonlinearities] - desired_y[:, -number_of_thetas:-number_of_nonlinearities]) ) \
     + tf.reduce_mean( 0.0*tf.square(predicted_y[:, -number_of_nonlinearities:] - desired_y[:, -number_of_nonlinearities:]) ) 


class pde_dnn_model:

    def __init__( self, number_of_inputs=0, number_of_outputs=0 ):
        
        self.set_input_output_dimensions( number_of_inputs, number_of_outputs )
        
        return
    
    def set_input_output_dimensions( self, number_of_inputs, number_of_outputs ):
        self.M_number_of_inputs  = number_of_inputs
        self.M_number_of_outputs = number_of_outputs
        return
    
    def construct_mlp_model( self, number_of_inputs, network_width='std_size' ):
        
        print( "MLP-network with network %s " % network_width )
        
        if network_width == 'std_encoder':
            model = keras.Sequential([
                    keras.layers.Dense( 1024, activation=tf.nn.relu, input_shape=(number_of_inputs,)),
                    keras.layers.Dense( 512, activation=tf.nn.relu),
                    keras.layers.Dense(256, activation=tf.nn.relu),
                    keras.layers.Dense(128, activation=tf.nn.relu) 
                    ])
        elif network_width == 'std_size':
            model = keras.Sequential([
                    keras.layers.Dense( 256, activation=tf.nn.relu, input_shape=(number_of_inputs,)),
                    keras.layers.Dense( 256, activation=tf.nn.relu),
                    keras.layers.Dense(256, activation=tf.nn.relu),
                    keras.layers.Dense(256, activation=tf.nn.relu)
                    ])
        elif network_width == 'small_size':
            model = keras.Sequential([
                    keras.layers.Dense( 64, activation=tf.nn.relu, input_shape=(number_of_inputs,)),
                    keras.layers.Dense( 64, activation=tf.nn.relu),
                    keras.layers.Dense(64, activation=tf.nn.relu),
                    keras.layers.Dense(64, activation=tf.nn.relu)
                    ])
        elif network_width == 'large_size':
            model = keras.Sequential([
                    keras.layers.Dense( 1024, activation=tf.nn.relu, input_shape=(number_of_inputs,)),
                    keras.layers.Dense( 1024, activation=tf.nn.relu),
                    keras.layers.Dense(1024, activation=tf.nn.relu),
                    keras.layers.Dense(1024, activation=tf.nn.relu)
                    ])
        elif network_width == 'long_large_size':
            model = keras.Sequential([
                    keras.layers.Dense( 1024, activation=tf.nn.relu, input_shape=(number_of_inputs,)),
                    keras.layers.Dense( 1024, activation=tf.nn.relu),
                    keras.layers.Dense( 1024, activation=tf.nn.relu),
                    keras.layers.Dense( 1024, activation=tf.nn.relu),
                    keras.layers.Dense( 1024, activation=tf.nn.relu),
                    keras.layers.Dense( 1024, activation=tf.nn.relu),
                    keras.layers.Dense( 1024, activation=tf.nn.relu),
                    keras.layers.Dense( 1024, activation=tf.nn.relu)
                    ])

            # the only case where the final number of mlp layers differs from 5
            self.M_number_of_mlp_layers = 9
        
        return model

    def build_model( self, network_width='std_size', restore_model=False, model_name='default_model_name' ):

        # default value for the loss function
        chosen_loss='mse'
        
        # building mlp part of the network        
        if restore_model == True:
            self.restore_pde_dnn_model( model_name )
            return
        else:
            if self.M_number_of_inputs == 0 or self.M_number_of_outputs == 0:
                print( '\n !! Warning, the number of input and outpus are set to %d and %d ' % ( self.M_number_of_inputs, self.M_number_of_outputs ) )
                assert( False )
                
            self.M_model = self.construct_mlp_model( self.M_number_of_inputs, network_width )

        # if we don't want the RB activation
        if self.M_use_rb_activation == False:
            self.M_model.add(tf.keras.layers.Dense( self.M_number_of_outputs ) )
            
        # instead, we add a layer outputting the number of theta functions (or parameter) followed by a sigmoid and the pde_mu_solver
        # which must have previously set
        else:
            chosen_loss = partial( custom_loss, 0.0, self.M_number_of_thetas, self.M_number_of_nonlinearities )

            self.M_model.add(tf.keras.layers.Dense( self.M_number_of_thetas, activation=tf.nn.sigmoid ) )
            self.add_rb_layer( )

        optimizer = tf.train.AdamOptimizer( learning_rate=self.M_learning_rate )

        self.M_model.compile(loss=chosen_loss,
                      optimizer=optimizer,
                      metrics=['mae'])

        return

    def add_rb_layer( self ):

        self.M_model.add(tf.keras.layers.Lambda( self.M_pde_activation.pde_mu_solver, \
                                                 output_shape=( self.M_number_of_outputs, ), \
                                                 input_shape=( self.M_number_of_thetas, ) ) )
        return
        

    def train_model( self, X_train, y_train, EPOCHS ):

        self.M_history = self.M_model.fit(X_train, y_train, epochs=EPOCHS, \
                         validation_split=0.2, verbose=1, \
                         callbacks=[PrintDot()], batch_size=self.M_batch_size )

        return

    def restore_pde_dnn_model( self, model_name ):
        
        chosen_loss='mse'
        self.M_model = tf.keras.models.load_model( model_name )

        if self.M_use_rb_activation == True:        
            chosen_loss = partial( custom_loss, 0.0, self.M_number_of_thetas, self.M_number_of_nonlinearities )
            self.add_rb_layer( )
            
        optimizer = tf.train.AdamOptimizer( learning_rate=self.M_learning_rate )

        self.M_model.compile(loss=chosen_loss,
                      optimizer=optimizer,
                      metrics=['mae'])

        return


    def save_pde_dnn_model( self, model_name ):
        
        if self.M_use_rb_activation == False:
            self.M_model.save( model_name, overwrite=True, include_optimizer=False )
        else:
            additional_pde_model = keras.Sequential( )

            for ii in range( self.M_number_of_mlp_layers  ):
                additional_pde_model.add( self.M_model.layers[ii] )

            additional_pde_model.save( model_name, overwrite=True, include_optimizer=False )

        return

    def predict( self, X_test ):
        
        test_predictions = self.M_model.predict( X_test )
        
        return test_predictions

    def evaluate_model_errors( self, X_test, y_test, evaluate_theta_errors=False ):
        
        test_predictions = self.predict( X_test )
        
        model_errors = abs( test_predictions - y_test )
        fem_output_coordinates = model_errors.shape[1] - self.M_number_of_thetas
        error_test = np.mean( model_errors[:, 0:fem_output_coordinates] )

        print( self.M_number_of_thetas )
        
        print( 'Errors on test is %f ' % error_test )

        if evaluate_theta_errors == True:
            error_test_parameters = model_errors[:, -self.M_number_of_thetas:]

            for iP in range( self.M_number_of_thetas ):
                error_test_parameters[self.M_number_of_thetas-iP-1] = np.mean( np.sqrt( model_errors[:, -self.M_number_of_thetas+iP] * model_errors[:, -(iP+1)] ) )
                print( "Error on parameter %d is %f " % ( iP, error_test_parameters[iP] ) )

        return error_test

    def plot_theta_predictions( self, X_test, y_test ):
        
        test_predictions = self.predict( X_test )
        model_errors = abs( test_predictions - y_test )
        fem_output_coordinates = model_errors.shape[1] - self.M_number_of_thetas

        plt.figure( )
    
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots( 2, 2, figsize=(15,15) )
    
        for iP in range( self.M_number_of_thetas ):
            ax1.scatter( y_test[:,fem_output_coordinates+iP] , test_predictions[:,fem_output_coordinates+iP], \
                         label='param_' + str(iP) )
    
        for iP in range( self.M_number_of_thetas ):
            ax2.scatter( y_test[:,fem_output_coordinates+iP] , \
                         np.abs( test_predictions[:,fem_output_coordinates+iP] - y_test[:,fem_output_coordinates+iP]), \
                         label='param_' + str(iP) )
    
        ax1.set_title( self.M_simulation_name + ' results y_test vs y_pred - A' )
        ax2.set_title( self.M_simulation_name + ' errors  |y_pred-y_test| vs y_pred ' )
        ax1.legend( )
        ax2.legend( )
    
        return

    def set_learning_rate( self, learning_rate ):
        self.M_learning_rate = learning_rate
        return 

    def set_use_rb_activation( self, use_rb_activation ):
        self.M_use_rb_activation = use_rb_activation
        return 

    def set_number_of_thetas( self, number_of_thetas ):
        self.M_number_of_thetas = number_of_thetas
        return
        
    def set_number_of_nonlinearities( self, number_of_nonlinearities ):
        self.M_number_of_nonlinearities = number_of_nonlinearities
        return
        
    def set_rb_activation( self, pde_activation ):
        self.M_pde_activation = pde_activation
        return        

    def set_simulation_name( self, simulation_name ):
        self.M_simulation_name = simulation_name
        return
    
    M_model = None
    M_pde_activation = None
    M_use_rb_activation = False
    M_number_of_inputs  = 0
    M_number_of_outputs = 0
    M_number_of_thetas  = 0         # it accounts for both linear and nopnlinear terms
    M_number_of_nonlinearities = 0  # < M_number_of_thetas. It accounts for those terms in M_number_of_thetas specific for the nonlinearities
    M_number_of_mlp_layers     = 5
    
    M_learning_rate = 0.001
    M_batch_size = 128
    M_history = None

    M_simulation_name = ""




