#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:07:28 2019

@author: Niccolo' Dal Santo
@email : niccolo.dalsanto@epfl.ch
"""

import random
import numpy as np

# generate the coordinates randomly or in a tensorial way (if the mesh is structured and the fem dofs are ordered in a certain way)

def generate_fem_coordinates( number_of_fem_coordinates, min_coord, max_coord, sampling='random', dof_per_direction=0, \
                              possible_coordinates=[] ):
    
    if sampling == 'random':

        fem_locations = np.zeros( number_of_fem_coordinates )
   
        if len(possible_coordinates)==0 :
            for iCoord in range( number_of_fem_coordinates ):
                random.seed(9011 * (iCoord + 1) + iCoord + 2)
                fem_locations[ iCoord ] = random.randint(min_coord,max_coord)
        elif len(possible_coordinates) > 0:
            for iCoord in range( number_of_fem_coordinates ):
                random.seed(9011 * (iCoord + 1) + iCoord + 2)
                fem_locations[ iCoord ] = possible_coordinates[random.randint(0,possible_coordinates.shape[0]-1)]
            
    elif sampling == 'tensorial':
        
        fem_locations = np.zeros( number_of_fem_coordinates[0] * number_of_fem_coordinates[1] )

        jump_x = np.ceil( float(dof_per_direction) / float( number_of_fem_coordinates[0] + 1 ) )
        jump_from_border_x = np.floor( ( float(dof_per_direction) - jump_x * float( number_of_fem_coordinates[0] + 1 ) ) / 2. )

        print( jump_from_border_x )

        jump_y = np.ceil( float(dof_per_direction) / float( number_of_fem_coordinates[1] + 1 ) )
        jump_from_border_y = np.floor( ( float(dof_per_direction) - jump_y * float( number_of_fem_coordinates[1] + 1 ) ) / 2. )
        
        print('Choosing tensorial grid selection, with jumps %f, %f and jumps from border %f, %f' \
            % (jump_x, jump_y, jump_from_border_x, jump_from_border_y) )
        
        fem_location_counter = 0;
        
        for iX in range( number_of_fem_coordinates[0] ):
            for iY in range( number_of_fem_coordinates[1] ):
                
                fem_locations[fem_location_counter] = jump_from_border_x  + dof_per_direction * jump_from_border_y \
                                                    + (iY+1) * jump_y * dof_per_direction \
                                                    + (iX+1) * jump_x
                
                fem_location_counter = fem_location_counter + 1
    
    fem_locations = np.sort( fem_locations )
    fem_locations = np.unique( fem_locations )
    
    return fem_locations.astype( int )



def generate_fem_training_data( ns, fem_coordinates, fem_output_coordinates, snapshot_collector ):

    num_locations = fem_coordinates.shape[0]
    num_output_locations = fem_output_coordinates.shape[0]

    y_output = np.zeros( ( ns, num_output_locations ) )

    # measurements of the solution, should they be noised?
    u_ex_locations = np.zeros( (ns, num_locations) )

    for iNs in range( ns ):
        u_ex_locations[iNs, :] = snapshot_collector.get_snapshot_function( iNs, fem_coordinates )
        y_output[iNs, :] = snapshot_collector.get_snapshot_function( iNs, fem_output_coordinates )

    return u_ex_locations, y_output




