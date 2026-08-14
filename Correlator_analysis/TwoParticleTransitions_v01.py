import numpy as np 

from scipy.integrate import quad

import sys 
sys.path.append( "/Users/markbook/Cobra" ) 

MASS=0.07329558992877808

M2   = MASS * MASS 

def compute_cm_q_momentum( s: float ) -> np.complex128: 
        return np.sqrt( 0.25 * s - M2 + 0.j ) 
    
def compute_one_particle_energy( momentum: float ) -> float: 
    return np.sqrt( momentum*momentum + M2 )  

def compute_scalar_spacetime_product( vec1: np.matrix, vec2: np.matrix )->float: 
    g = np.matrix( [ [1,0], [0,-1] ] ) 
    return (vec1 @ g @ vec2.T)[0,0] 

class Kinematics_in_FV( object ): 

    def __init__( self, Pf: np.matrix, Pi: np.matrix, LEN: int, XI=0.5, MASS=0.073295 ): 

        self.Pf   = Pf    ## Momentum of the final two-particle state
        self.Pi   = Pi    ## Momentum of the initial two-particle state
        self.MASS = MASS  ## Mass of the field 
        self.LEN  = LEN   ## Size of the volume  
        self.XI   = XI    ## Symetry factor of s-loop 

        self.M2   = MASS * MASS 
        self.UNIT = 2. * np.pi / LEN 

        self.sf   = compute_scalar_spacetime_product( Pf, Pf ) 
        self.si   = compute_scalar_spacetime_product( Pi, Pi ) 
        self.Q2   = - compute_scalar_spacetime_product( Pf-Pi, Pf-Pi ) 
        self.qf   = compute_cm_q_momentum( self.sf )
        self.qi   = compute_cm_q_momentum( self.si ) 
        self.PfPi = compute_scalar_spacetime_product( Pf, Pi )

        pass

    def compute_cm_q_momentum( self, s: float ) -> np.complex128: 
        return np.sqrt( 0.25 * s - self.M2 + 0.j ) 
    
    def compute_one_particle_energy( self, momentum: float ) -> float: 
        return np.sqrt( momentum*momentum + self.M2 )  
    
    def compute_scalar_spacetime_product( self, vec1: np.matrix, vec2: np.matrix )->float: 
        g = np.matrix( [ [1,0], [0,-1] ] ) 
        return (vec1 @ g @ vec2.T)[0,0] 

    def make_Lorentz_transform( self, reference: np.matrix, vec_original: np.matrix )->np.matrix: 

        gamma = reference[0,0] / np.sqrt( self.compute_scalar_spacetime_product( reference, reference ) ) 
        beta  = reference[0,1] / reference[0,0] 
    
        vec_new_0 = gamma * ( vec_original[0,0] - beta * vec_original[0,1] )
        vec_new_1 = gamma * ( vec_original[0,1] - beta * vec_original[0,0] ) 

        return np.matrix( [ vec_new_0, vec_new_1 ] ) 
    
    def compute_phase_space( self, s: float ) -> float:  

        q_cm = self.compute_cm_q_momentum( s ) 

        return 0.25 * self.XI / ( np.sqrt( s ) * q_cm ) 
    
    def compute_F_function( self, P: np.matrix, d: int ) -> np.complex128: 

        s     = self.compute_scalar_spacetime_product( P, P ) 
        q_cm  = self.compute_cm_q_momentum( s ) 
        gamma = P[0,0] / np.sqrt( s ) 

        rho  = self.compute_phase_space( s ) 

        return rho * ( 1.j + 1. / np.tan( 0.5 * gamma * q_cm * self.LEN + 0.5 * np.pi * d ) ) 
    
    def compute_g_star_L( self, df: int ) -> np.complex128: 

        Q2_pos, Q2_neg = self.compute_roots_in_Q2() 

        FPf = self.compute_F_function( self.Pf, df ) 

        numerator_1    = 0.5 * ( self.si - self.sf - self.Q2 )
        denominator_1  = ( self.Q2 - Q2_pos ) * ( self.Q2 - Q2_neg ) 
        factor_1       = numerator_1 / denominator_1

        numerator_2    = - self.sf * FPf 
        denominator_2  = self.XI * self.M2 
        factor_2       = numerator_2 / denominator_2

        return factor_1 * factor_2
    
    def compute_g_bullet_L( self, di: int ) -> np.complex128: 

        Q2_pos, Q2_neg = self.compute_roots_in_Q2() 

        FPi = self.compute_F_function( self.Pi, di ) 

        numerator_1    = 0.5 * ( self.sf - self.si - self.Q2 )
        denominator_1  = ( self.Q2 - Q2_pos ) * ( self.Q2 - Q2_neg ) 
        factor_1       = numerator_1 / denominator_1

        numerator_2    = - self.si * FPi 
        denominator_2  = self.XI * self.M2 
        factor_2       = numerator_2 / denominator_2

        return factor_1 * factor_2
    
    def compute_scalar_GL_contribution( self, df: int, di: int ) -> np.complex128: 

        gL_star   = self.compute_g_star_L( df ) 
        gL_bullet = self.compute_g_bullet_L( di ) 

        return gL_star + gL_bullet
    
    def compute_aux_vector_V_f( self ) -> np.matrix:  

        Pi_star = self.make_Lorentz_transform( self.Pf, self.Pi )

        V0_cm = ( self.si - self.PfPi ) * np.sqrt( self.sf ) 
        V1_cm = - 4. * Pi_star[0,1] * self.qf * self.qf 
        V_cm  = np.matrix( [ V0_cm, V1_cm ] )
        V_on  = self.make_Lorentz_transform( np.matrix( [ self.Pf[0,0], -self.Pf[0,1] ] ), V_cm )

        return V_on 
    
    def compute_aux_vector_V_i( self ) -> np.matrix: 

        Pf_bullet = self.make_Lorentz_transform( self.Pi, self.Pf ) 

        V0_cm = ( self.sf - self.PfPi ) * np.sqrt( self.si ) 
        V1_cm = - 4. * Pf_bullet[0,1] * self.qi * self.qi 
        V_cm  = np.matrix( [ V0_cm, V1_cm ] ) 
        V_on  = self.make_Lorentz_transform( np.matrix( [ self.Pi[0,0], -self.Pi[0,1] ] ), V_cm )

        return V_on 
    
    def compute_vector_GL_contribution( self, df, di ) -> np.matrix: 

        Vf_on = self.compute_aux_vector_V_f() 
        Vi_on = self.compute_aux_vector_V_i() 

        coeff_f = self.compute_g_star_L( df ) / ( self.si - self.sf - self.Q2 ) 
        coeff_i = self.compute_g_bullet_L( di ) / ( self.sf - self.si - self.Q2 ) 

        return coeff_f * Vf_on + coeff_i * Vi_on
    
    ###### OLD CODE BELOW  ####
    ###### OLD CODE BELOW  ####

    
    def compute_roots_in_Q2( self ) -> tuple:  

        sf, si = self.sf, self.si 
        qf, qi = self.qf, self.qi 

        term1  = .5 * si * sf / self.M2 - si - sf 
        term2  = 2. * np.sqrt( si * sf ) * qf * qi / self.M2

        positive_root = term1 + term2 
        negative_root = term1 - term2 

        return ( positive_root, negative_root ) 
    
    def compute_Q2_polynom( self ) -> np.complex128: 

        Q2 = self.Q2 
        Q2_pos, Q2_neg = self.compute_roots_in_Q2() 

        return 2. * self.M2 * self.XI * ( Q2 - Q2_pos ) * ( Q2 - Q2_neg ) 
    
    def compute_phase_space( self, s: float ) -> np.complex128: 

        q = self.compute_cm_q_momentum( s ) 

        return 0.25 * self.XI / ( q * np.sqrt( s ) ) 
    
    def compute_onshell_momenta( self, P: np.matrix ) -> np.matrix: 

        P_boost_inv = np.matrix( [ P[0,0], -P[0,1] ] )

        s    = self.compute_scalar_spacetime_product( P, P ) 
        q_cm = self.compute_cm_q_momentum( s )

        k_cm = np.matrix( [ 0.5*np.sqrt(s), q_cm ] ) 

        k_on = self.make_Lorentz_transform( P_boost_inv, k_cm )

        return k_on 
    
    ###### OLD CODE ABOVE  ####
    ###### OLD CODE ABOVE  ####
    
class Kinematics( object ): 

    def __init__( self, Pf: np.matrix, Pi: np.matrix, EPS: float, XI=0.5, MASS=0.073295 ): 

        self.Pf   = Pf    ## Momentum of the final two-particle state
        self.Pi   = Pi    ## Momentum of the initial two-particle state
        self.MASS = MASS  ## Mass of the field 
        self.EPS  = EPS   ## Size of the volume  
        ## THIS EPSILON PRESCRIPTION IS CONSTANT 
        ## THE ONE IN THE DISSERTATION DEPENDS ON 
        ## THE FEYNMAN PARAMETER x
        self.XI   = XI    ## Symetry factor of s-loop 

        self.M2   = MASS * MASS 

        self.sf   = compute_scalar_spacetime_product( Pf, Pf ) 
        self.si   = compute_scalar_spacetime_product( Pi, Pi ) 
        self.Q2   = - compute_scalar_spacetime_product( Pf-Pi, Pf-Pi ) 
        self.qf   = compute_cm_q_momentum( self.sf )
        self.qi   = compute_cm_q_momentum( self.si ) 
        self.PfPi = compute_scalar_spacetime_product( Pf, Pi )

        pass 

    def compute_polynomial_A_of_x( self, x: float ) -> float: 

        return 1 - 2 * self.PfPi * x / self.si 
    
    def compute_polynomial_B_of_x( self, x: float ) -> float: 

        return -4. * ( self.M2 - x * ( 1 - x) * self.sf )  / self.si 
    
    def compute_roots_of_polynomial_in_y( self, x: float ) -> np.ndarray: 

        A_of_x = self.compute_polynomial_A_of_x( x ) 
        B_of_x = self.compute_polynomial_B_of_x( x ) 
        aux    = np.sqrt( A_of_x*A_of_x + B_of_x + 0.j )

        return np.array( [ 0.5 * ( A_of_x + aux ), 0.5 * ( A_of_x - aux ) ] ) 
    
    def compute_Ln_function( self, x: float, EPS:float, order_n=1 ) -> np.array:  

        y_roots      = self.compute_roots_of_polynomial_in_y( x ) 
        #y_roots_weps = y_roots + np.array( [ 1.j*self.EPS, -1.j*self.EPS ] )
        y_roots_weps = np.array( [ y_roots[0]+1.j*EPS, y_roots[1]-1.j*EPS ] )

        if order_n == 1: 

            numerator1   = 1. - x - y_roots 
            denominator1 = y_roots 
            term1        = np.log( np.abs( numerator1 / denominator1 ) ) 

            numerator2   = y_roots_weps.imag 
            denominator2 = 1. - x - y_roots_weps.real 
            term2        = -1.j * np.arctan( numerator2 / denominator2 ) ## CHECK THAT THIS IS MAPPED TO THE CORRECT QUADRANT!! 

            numerator3   = y_roots_weps.imag
            denominator3 = y_roots_weps.real 
            term3        = 1.j * np.arctan( numerator3 / denominator3 ) 

            ## THE LOG FUNCTION ALREADY HANDLES COMPLEX NUMBERS 
            ## COMPUTING PHASES SEPARATELY IS REDUNDANT, COSTLY, 
            ## AND MAY LEAD TO MISTAKES. 

            ##return term1 + term2 + term3 

            return np.log((1-x-y_roots_weps)/-y_roots_weps)
        
        elif order_n == 2: 

            term1 = 1. / ( y_roots_weps - ( 1. - x ) ) 
            term2 = - 1. / y_roots_weps 

            return term1 + term2 
        
    def compute_gn_function( self, x: float, EPS: float, order_n=1 ) -> np.complex128: 

        L_functions = self.compute_Ln_function( x, EPS, order_n=order_n ) 
        y_roots     = self.compute_roots_of_polynomial_in_y( x )

        numerator   = 0.25 * ( L_functions[0] + (-1)**order_n * L_functions[1] )
        denominator = np.pi * self.si * self.si * np.power( y_roots[0] - y_roots[1] + 2.j*EPS, 4-order_n )

        return numerator / denominator
    
    def compute_scalar_G_contribution( self, EPS: float ) -> tuple: 

        def integrand(x): 

            g1 = self.compute_gn_function( x, EPS, order_n=1 ) 
            g2 = self.compute_gn_function( x, EPS, order_n=2 ) 

            return g2 - 2. * g1 
        
        real_part = quad( lambda x: np.real( integrand(x) ), 0., 1., limit=200 ) 
        imag_part = quad( lambda x: np.imag( integrand(x) ), 0., 1., limit=200 )  

        return ( real_part[0] + 1.j * imag_part[0] ), ( real_part[1] + 1.j * imag_part[1] ) 
    
    def compute_h2_function( self, x: float, EPS: float ) -> np.complex128: 

        L2_functions = self.compute_Ln_function( x, EPS, order_n=2 ) 
        y_roots      = self.compute_roots_of_polynomial_in_y( x ) 

        factor = 0.25 / ( np.pi * self.si * self.si ) 

        numerator   = y_roots[0] * L2_functions[0] + y_roots[1] * L2_functions[1] 
        denominator = np.power( y_roots[0] - y_roots[1] + 2.j * self.EPS, 2 ) 

        return factor * numerator / denominator 
    
    def compute_vector_If_integral( self, EPS: float ) -> np.complex128: 

        def integrand( x ): 

            g1 = self.compute_gn_function( x, EPS, order_n=1 ) 
            g2 = self.compute_gn_function( x, EPS, order_n=2 ) 

            return x * ( g2 - 2. * g1  ) 
        
        real_part = quad( lambda x: np.real( integrand(x) ), 0., 1., limit=200 ) 
        imag_part = quad( lambda x: np.imag( integrand(x) ), 0., 1., limit=200 ) 

        return ( real_part[0] + 1.j * imag_part[0] ), ( real_part[1] + 1.j * imag_part[1] ) 
    
    def compute_vector_Ii_integral( self, EPS: float ) -> np.complex128: 

        def integrand( x ): 

            g1 = self.compute_gn_function( x, EPS, order_n=1 ) 
            h2 = self.compute_h2_function( x, EPS ) 

            y_roots = self.compute_roots_of_polynomial_in_y( x ) 

            return h2 - self.compute_polynomial_A_of_x( x ) * g1
        
        real_part = quad( lambda x: np.real( integrand(x) ), 0., 1., limit=200 ) 
        imag_part = quad( lambda x: np.imag( integrand(x) ), 0., 1., limit=200 ) 

        return ( real_part[0] + 1.j * imag_part[0] ), ( real_part[1] + 1.j * imag_part[1] ) 
    
    def compute_vector_G_contribution( self, EPS: float, mu=0 ) -> tuple: 

        Pf_mu, Pi_mu = self.Pf[0,mu], self.Pi[0,mu] 

        If_mean, If_err = self.compute_vector_If_integral( EPS ) 
        Ii_mean, Ii_err = self.compute_vector_Ii_integral( EPS )  

        Gmu_mean = ( Pf_mu * If_mean + Pi_mu * Ii_mean )
        Gmu_err  = ( Pf_mu * If_err  + Pi_mu * Ii_err  ) 

        return Gmu_mean, Gmu_err


    




