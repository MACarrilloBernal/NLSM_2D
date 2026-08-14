import numpy as np 
import scipy.linalg as lin 

#import pathlib as Path 

import sys
sys.path.append( "/Users/markbook/Cobra" ) 
import jackknife as jk 

#from pprint import pprint 
from iminuit import Minuit 

from pathlib import Path 

## An integral library for the generation of field configurations and 
## the analysis of their correlation functions. The current version 
## handles a 'float' or 'np.ndarray' type variable of 's' and a single 
## value for the mass 'm' of the field. Future versions will handle an 
## ensemble of mass values.  For this, a subroutine that identifies and 
## handles the type of input variables ( 's' and 'm' ) is necessary. 

## s - invariant mass of the two-particle system, may be single valued or 
##     a range of values 

## Analytic results of the NLSM 
class NLSM( object ): 

    ## INITIALIZER for NLSM
    def __init__( self, TIME=256, LEN=128, COUPLING=1.54, MASS=0.073295 ):

        self.TIME     = TIME      ## extension of time dimension 
        self.LEN      = LEN       ## extension of space dimension 
        self.COUPLING = COUPLING  ## coupling constant 

        self.UNIT     = 2.0 * np.pi / LEN  ## unit of momentum 
 
        self.MASS     = MASS  ## Reassigned from 'one_2pt' analysis 

        self.M2       = MASS*MASS 

        pass

    ## CM frame rapidity of one particle 
    def rapidity( self, s: np.ndarray ) -> np.ndarray: 

        ## Magnitude of the relative momentum of one particle in the 
        ## CM frame in units of the mass 
        k_by_m = np.sqrt( 0.25 * s / self.M2 - 1.0 )  

        return 2.0 * np.arcsinh( k_by_m )
    
    ## 1+1D phsae-space 
    def phase_space( self, s: np.ndarray, xi=0.5 ) -> np.ndarray: 

        q = np.sqrt( 0.25 * s - self.M2 )

        return .25 * xi / ( np.sqrt( s ) * q ) 
    
    def K_matrix_inv( self, s: np.ndarray ) -> np.ndarray: 

        theta = self.rapidity( s ) 
        rho   = self.phase_space( s ) 

        return rho * ( theta / np.pi + 2.0 * np.pi / theta ) 
    
    ## One-particle form-factor analytic (K&W) 
    def form_factor_1P( self, s ): 

        theta   = self.rapidity( s ) 
        factor1 = theta / ( 2.0 * np.tanh( 0.5 * theta ) )  
        factor2 = np.square( np.pi ) / ( np.square( np.pi ) + np.square( theta ) ) 

        return factor1 * factor2

    ## Phase-shift of the two-particle isovector scattering channel 
    def phase_shift_I1( self, s: np.ndarray ) -> np.ndarray: 

        theta       = self.rapidity( s ) 
        numerator   = ( theta + 2.0j*np.pi ) * ( theta - 1.0j*np.pi ) 
        denominator = ( theta - 2.0j*np.pi ) * ( theta + 1.0j*np.pi ) 

        return -0.5j * np.log( numerator / denominator ) 
    
    ## Polynomial function of 's' for the quantization condition 
    def polynom_in_s( self, s: np.ndarray, BOOST: int ) -> np.ndarray: 

        ## Squared total momentum of the two-particle system 
        P2 = np.square( BOOST * self.UNIT ) 

        #return ( s + P2 ) * ( s - 4.0*self.M2 ) 
        return ( s + P2 ) * ( s - 4.0*self.MASS*self.MASS ) 
    
    ## Cut function of the polynom above for the quantization condition 
    ## 'n' indicates the label of the energy level, and 'delta' is the 
    ## scattering phase-shift evaluated at 's'. The non-interacting 
    ## case is assumed by default 
    def cut_in_s( self, s: np.ndarray, n: int, BOOST: int, 
                  delta=0.0 ) -> np.ndarray: 

        L2  = self.LEN * self.LEN 
        eta = BOOST % 2  ## Indicates the parity of the boost 

        return np.square( (2*n+eta)*np.pi - 2.0*delta ) * 4.0*s / L2  
    
    ## Search the n-th energy level of a given boost within the range 
    ## of 's' 
    def search_level( self, s: np.ndarray, n: int, BOOST: int, 
                      phase_shift_fun, steps=1000, FREE=False, 
                      delta=0.0 ) -> tuple:  
        
        if not FREE: delta = phase_shift_fun( s ).real

        pol_s = self.polynom_in_s( s, BOOST )
        cut_s = self.cut_in_s( s, n, BOOST, delta=delta ) 
        diff  = pol_s - cut_s 

        check_sign = np.nonzero( np.diff( np.sign( diff ) ) )[0]

        if len( check_sign ): 
            pointer = check_sign[0] 
            if pointer == 0: 
                return ()             
            else: 
                E_level = np.sqrt( 0.5 * ( s[pointer-1]+s[pointer+1] ) ) 
                return ( pointer, E_level )
        else: 
            return () 
    
    ## Apply the routine above in a finner range of 's' 
    def refine_level( self, old_s: np.ndarray, pointer: int, n: int, 
                      BOOST: int, phase_shift_fun, steps=1000, 
                      FREE=False, delta=0.0 ) -> tuple: 
        
        ## step size in s
        ds = ( old_s[pointer+1] - old_s[pointer-1] ) / steps 

        ## new finner range of s
        new_s = np.arange( old_s[pointer-1], old_s[pointer+1]+ds, ds ) 

        if not FREE: delta = phase_shift_fun( new_s ).real 

        return new_s, self.search_level( new_s, n, BOOST, phase_shift_fun, steps=steps, FREE=FREE ) 
    
    ## Compute the two-particle spectrum for a given boost within a 
    ## range of 's' and either a given scattering channel or a free 
    ## theory. A default precission is set but cannot be taken too low 
    ## in the current implementation. 
    def get_spectrum( self, s: np.ndarray, BOOST: int, phase_shift_fun, 
                      ref_err=5E-10, steps=100, n_max=5, FREE=False )->list: 
        
        levels = [] 

        for n in range( 0, n_max+1, 1 ): 

            ## search the n-th energy level 
            search = self.search_level( s, n, BOOST, phase_shift_fun, 
                                        steps=steps, FREE=FREE ) 
            
            ## refine the value of the energy level, if any 
            if search: 

                new_s = s
                rel_err = 1.0 
                pointer, level = search

                while rel_err > ref_err: 
                    old_level = level 
                    new_s, search = self.refine_level( new_s, pointer, n, 
                                                       BOOST, phase_shift_fun, 
                                                       steps=steps, FREE=FREE ) 
                    pointer, level = search
                    rel_err = np.abs( ( level - old_level ) / ( level + old_level ) ) 

                levels.append( ( level, n ) )

            ## if no level is gotten, skip label 'n'
            else:
                levels.append( () ) 

        return levels 

    def phase_shift_L( self, s, boost ): 

        P2    = np.square( boost * self.UNIT ) 
        gamma = np.sqrt( ( s+P2 ) / s ) 
        qcm   = np.sqrt( 0.25 * s - self.M2 )  

        return 0.5 * ( gamma*qcm*self.LEN - boost*np.pi ) 
    
    def K_inv_ERE( self, s, pars ): 

        A, R = pars[0], pars[1] ## -1/a, r/2 

        rho = self.phase_space( s ) 
        qcm = np.sqrt( 0.25*s - self.M2 ) 

        return rho * ( A/qcm + R*qcm ) 
    
    def phase_shift_ERE( self, s, pars ): 

        K_inv = self.K_inv_ERE( s, pars ) 
        rho   = self.phase_space( s ) 

        return np.arctan( rho / K_inv ) 
    
    def ddE_phase_shift_L( self, s, boost ): 

        P2  = np.square( boost * self.UNIT ) 
        q2  = 0.25*s - self.M2 

        return 0.125 * self.LEN * ( s + 4.0*self.M2*P2/s ) / np.sqrt( q2*s )
    
    def ddR_phase_shit_ERE( self, s, boost, pars ): 

        a, r = -1.0/pars[0], 2.0*pars[1] 

        P2  = np.square( boost * self.UNIT ) 
        E   = np.sqrt( s + P2 ) 
        qcm = np.sqrt( 0.25*s - self.M2 ) 
        eta = 2.0*a*qcm / ( a*r*qcm*qcm - 2.0 ) 

        return 0.25*r*E*eta * ( 1.0/(r*qcm) - eta )/( qcm*( 1.0+eta*eta ) ) 
    
    def get_LLFactor_inv_ERE( self, s, boost, pars ): 

        rho   = self.phase_space( s ) 
        phase = np.exp( -2.0j * self.phase_shift_L( s, boost ) ) 

        ddE_delta_ERE = self.ddR_phase_shit_ERE( s, boost, pars ) 
        ddE_delta_L   = self.ddE_phase_shift_L( s, boost ) 

        return phase * ( ddE_delta_ERE + ddE_delta_L ) / rho 
    
def file_to_jk( path: str )->np.ndarray: 

    data = []
    for line in open( path, "r" ): 
        data.append( np.fromstring( line, dtype=float, sep=" " ) ) 
    data = np.array( data ).T 

    return np.array([ jk.quickknife( x ) for x in data ])
        
class one_2pt( object ): 

    def __init__( self, NLSM ):

        self.NLSM = NLSM 

    def read_corrs( self, paths: list, im_paths=False )->None: 

        self.re_corrs_jk = []  
        self.im_corrs_jk = [] 

        for i in range( len( paths ) ): 
            self.re_corrs_jk.append( file_to_jk( paths[i] ) )
            if im_paths: 
                self.im_corrs_jk.append( file_to_jk( im_paths[i] ) )

    def eff_masses( self, SHIFT=1 )->None: 

        self.eff_ms_jk = [] 
        for data_jk in self.re_corrs_jk:
            m_jk = np.log( data_jk[:-SHIFT] / data_jk[SHIFT:] ) / SHIFT 
            self.eff_ms_jk.append( m_jk ) 

    def make_fit( self, ti: int, tf: int, STATE: int, 
                  GUESS=[0.5,100], CHI2dof=False )->tuple: 

        t_fit = np.arange( ti, tf+1 ) 

        def hypth( pars, t ): 
            return pars[1] * np.exp( - pars[0] * t ) 
        
        dof     = len( t_fit ) - 2
        E_guess = GUESS[0] 
        A_guess = GUESS[1]

        ensemble_jk = self.re_corrs_jk[ STATE ][ ti : tf+1 ].real
        cov_matrix  = jk.jk_cov_matrix( ensemble_jk ) 
        InvCov      = np.matrix( np.linalg.inv( cov_matrix ) ) 

        E_jk, A_jk = [], [] 

        if CHI2dof: Chi2norm_jk = [] 

        for data in ensemble_jk.T : 

            def Chi2( E, A ): 

                fit   = hypth( [E,A], t_fit ) 
                diffs = np.matrix( data - fit ).T 

                return ( diffs.T @ InvCov @ diffs )[0,0] 
            
            Chi2_fit = Minuit( Chi2, E=E_guess, A=A_guess ) 

            Chi2_fit.errordef = Minuit.LEAST_SQUARES 
            Chi2_fit.strategy = 2 
            Chi2_fit.errors   = 0.000001 

            Chi2_fit.migrad() 

            E_jk.append( Chi2_fit.values[0] )
            A_jk.append( Chi2_fit.values[1] ) 

            if CHI2dof:
                Chi2norm_jk.append( Chi2_fit.fval/dof ) 

        if not STATE:
            self.NLSM.MASS    = np.mean(  E_jk ) 
            self.NLSM.MASS_jk = np.array( E_jk )

        if CHI2dof:
            return np.array(E_jk), np.array(A_jk), np.array(Chi2norm_jk) 
        else:
            return np.array(E_jk), np.array(A_jk) 
        
    def fit_spectrum( self, fit_ranges: list, GUESS=False, 
                      COEFFS=True, CHI2dof=False )->None:
        
        nof_states = len( fit_ranges ) 

        if not np.any( GUESS ): GUESS = [ [0.5,100] ] * nof_states 

        self.spectrum = [] 
        if COEFFS: self.overlaps = []  
        if CHI2dof: self.Chi2norms = [] 

        for n in range( nof_states ): 

            ti, tf = fit_ranges[n] 

            aux = self.make_fit( ti, tf, n, GUESS=GUESS[n], CHI2dof=CHI2dof )

            self.spectrum.append( aux[0] )
            if COEFFS: self.overlaps.append( aux[1] ) 
            if CHI2dof: self.Chi2norms.append( aux[2] )

    def save_data( self, NOF_STATES=1, JK_COPIES=1000, CHI2dof=False, folder="./" )->None: 

        folder = folder+"Pars_for_C2pt/"

        Path( folder ).mkdir( parents=True, exist_ok=True ) 

        for n in range( NOF_STATES ): 
            file = open( folder+"/one_2pt_pars_n{0}.jk".format(n), "w" )
            if CHI2dof: 
                file.write( "E Z2 Chi2/dof for d={0}\n".format( n ) )
            else:
                file.write( "E Z2 for d={0}\n".format( n ) )
            for k in range( JK_COPIES ): 
                if CHI2dof:
                    file.write( "{0:.20f} {1:.20f} {2:.20f}\n".format( self.spectrum[n][k], self.overlaps[n][k], self.Chi2norms[n][k] ) )
                else:
                    file.write( "{0:.20f} {1:.20f}\n".format( self.spectrum[n][k], self.overlaps[n][k], self.Chi2norms[n][k] ) )
            file.close()

    def C2pt_from_atributes( self, state: int, tmax: int )->np.ndarray: 

        E_jk = self.spectrum[  state ] 
        A_jk = self.overlaps[ state ] 

        return np.array( [ A_jk * np.exp( - E_jk * t ) for t in range( tmax ) ] )

    def C2pt_from_file( self, path: str, tmax: int )->np.ndarray:  

        #pars_jk = file_to_jk( path ) ## WRONG, THIS DUPLICATES JACKKINFING
        pars_jk = [] 

        for line in open( path, "r" ): 

            pars_jk.append( np.fromstring( line, dtype=float, sep=" " ) ) 

        pars_jk = np.array( pars_jk ).T

        return np.array( [ pars_jk[1] * np.exp( - pars_jk[0] * t ) for t in range( tmax ) ] )

class one_3pt( object ): 

    def __init__( self, NLSM: NLSM, one_2pt: one_2pt, T_SEP: int ): 

        self.NLSM    = NLSM 
        self.one_2pt = one_2pt
        self.T_SEP   = T_SEP 

        self.re_3pt_corrs    = [] 
        self.im_3pt_corrs    = [] 
        self.corr_labels     = [] 
        self.formfactors     = [] 
        self.phases          = [] 
        self.Chi2            = [] 
        self.Q2              = [] 
        self.formfactor_fit  = [] 
        self.formfactor_pars = [] 

    def read_corrs( self, re_path, im_path, label ): 

        self.re_3pt_corrs.append( file_to_jk( re_path ) )
        self.im_3pt_corrs.append( file_to_jk( im_path ) )
        self.corr_labels.append( label ) 

    def compute_ratio( self, pointer, d_out, d_in, EXCHANGE=False ): 

        def aux_fun( t, C2pt_out, C2pt_in ):
                return np.sqrt( C2pt_out[2*(self.T_SEP-t)] * C2pt_in[2*(self.T_SEP+t)] )

        re_C3pt_jk = self.re_3pt_corrs[ pointer ]
        im_C3pt_jk = self.im_3pt_corrs[ pointer ] 

        abs_C3pt_jk = np.sqrt( np.square(re_C3pt_jk) + np.square(im_C3pt_jk) )

        if d_out != d_in : 

            E_out = self.one_2pt.spectrum[ np.abs( d_out ) ] 
            E_in  = self.one_2pt.spectrum[ np.abs( d_in )  ] 

            factor = 2.0 * np.sqrt( E_out * E_in ) / ( E_out + E_in ) 

            C2pt_out = self.one_2pt.C2pt_from_atributes( np.abs(d_out), 4*self.T_SEP ) 
            C2pt_in  = self.one_2pt.C2pt_from_atributes( np.abs(d_in),  4*self.T_SEP ) 

            #denom = [ aux_fun( t, C2pt_out, C2pt_in ) for t in range( -self.T_SEP+1, self.T_SEP, 1 ) ] 
            denom = [ np.sqrt( C2pt_out[2*(self.T_SEP-tc)] * C2pt_in[2*(self.T_SEP+tc)] ) for tc in range( -self.T_SEP+1, self.T_SEP ) ]

            if EXCHANGE:

                label      = self.corr_labels[ pointer ] 
                label_ex   = ( label[1], label[0] )
                pointer_ex = self.corr_labels.index( label_ex ) 

                re_C3pt_ex_jk = self.re_3pt_corrs[ pointer_ex ] 
                im_C3pt_ex_jk = self.im_3pt_corrs[ pointer_ex ] 

                abs_C3pt_ex_jk = np.sqrt( np.square(re_C3pt_ex_jk) + np.square(im_C3pt_ex_jk) ) 

                C2pt_out_ex = self.one_2pt.C2pt_from_atributes( np.abs(d_in),  4*self.T_SEP ) 
                C2pt_in_ex  = self.one_2pt.C2pt_from_atributes( np.abs(d_out), 4*self.T_SEP )  
                
                denom_ex = [ np.sqrt( C2pt_out[2*(self.T_SEP+tc)] * C2pt_in[2*(self.T_SEP-tc)] ) for tc in range( -self.T_SEP+1, self.T_SEP, 1 ) ]

                print(  )
                for nu in range( 2*self.T_SEP-1 ): 
                    print( denom[nu][0], denom_ex[nu][0] )
                print(  )

                mid_step = [ 0.5 * ( abs_C3pt_jk[nu] / denom[nu] + abs_C3pt_ex_jk[nu] / denom_ex[nu] ) for nu in range( 2*self.T_SEP-1 ) ]
                #mid_step = [ 0.5 * ( abs_C3pt_ex_jk[nu] / denom_ex[nu]  ) for nu in range( 2*self.T_SEP-1 ) ]

                return np.array( [ factor * mid_step[nu] for nu in range( 2*self.T_SEP-1 ) ] )

            else: 
                return np.array( [ factor * abs_C3pt_jk[nu] / denom[nu] for nu in range( 2*self.T_SEP-1 ) ] )

        else: 
            C2pt_jk = self.one_2pt.C2pt_from_atributes( d_out, 2*self.T_SEP+1 )[-1]

            return np.array( [ x / C2pt_jk for x in abs_C3pt_jk ] )

    def get_form_factor( self, label, EXCHANGE=False ): 

        pointer = self.corr_labels.index( label ) 

        d_out = np.abs( label[0] ) 
        d_in  = np.abs( label[1] )  

        E_out = self.one_2pt.spectrum[ d_out ]
        E_in  = self.one_2pt.spectrum[ d_in  ] 

        Q2 = np.square( ( label[0]-label[1] )*self.NLSM.UNIT ) - np.square( E_out - E_in ) 

        self.formfactors.append( self.compute_ratio( pointer, d_out, d_in, EXCHANGE=EXCHANGE ) )
        self.Q2.append( Q2 ) 
        #self.phases.append( np.arctan( im_C3pt_jk / re_C3pt_jk )  ) 

    def fit_form_factor( self, label, ti, tf, GUESS=0.95 ): 

        pointer    = self.corr_labels.index( label ) 

        nu_i, nu_f = ti + self.T_SEP-1, tf + self.T_SEP-1

        sample_jk = np.array( self.formfactors[pointer][ nu_i : nu_f+1 ] ).real 
        cov       = jk.jk_cov_matrix( sample_jk ) 
        InvCov    = np.matrix( np.linalg.inv( cov ) ) 

        A_jk, Chi2_jk = [], [] 

        dof = len( sample_jk ) - 1 

        for data in sample_jk.T : 

            def Chi2( A ): 
                diff = np.matrix( data - A ).T 
                return ( diff.T @ InvCov @ diff )[0,0] 
            
            Chi2_fit = Minuit( Chi2, A=GUESS ) 

            Chi2_fit.errordef = Minuit.LEAST_SQUARES 
            Chi2_fit.strategy = 2 
            Chi2_fit.errors   = 0.0001 

            Chi2_fit.migrad() 

            A_jk.append(    Chi2_fit.values[0] ) 
            Chi2_jk.append( Chi2_fit.fval / dof ) 

        self.formfactor_fit.append( np.array( A_jk ) )
        self.Chi2.append(        np.array( Chi2_jk ) )

    def fit_form_factor_in_Q2( self ):

        def hypth( Q2, mu2, A ): 
            return 1.0 / ( Q2/mu2 + 1.0 ) + A * Q2 
        
        samples_jk = np.array( self.formfactor_fit )  
        Q2_jk      = np.array( self.Q2 ) 
        dof        = len( Q2_jk ) - 2

        cov    = jk.jk_cov_matrix( samples_jk ) 
        InvCov = np.matrix( np.linalg.inv( cov ) )  

        mu2_jk  = [] 
        A_jk    = [] 
        Chi2_jk = [] 

        for nu, data in enumerate( samples_jk.T ): 

            Q2 = Q2_jk.T[nu] 

            def Chi2( mu2, A ): 
                diff = np.matrix( data - hypth( Q2, mu2, A ) ).T 
                return ( diff.T @ InvCov @ diff )[0,0] 
            
            ## GENERALIZE GUSSES
            Chi2_fit = Minuit( Chi2, mu2=0.236, A=2.10 ) 

            Chi2_fit.errordef = Minuit.LEAST_SQUARES
            Chi2_fit.strategy = 2
            Chi2_fit.errors   = 0.0001 

            Chi2_fit.migrad() 

            mu2_jk.append(  Chi2_fit.values[0] ) 
            A_jk.append(    Chi2_fit.values[1] ) 
            Chi2_jk.append( Chi2_fit.fval / dof )

        self.formfactor_pars = np.array( [ mu2_jk, A_jk ] ).T 

    def save_data( self, folder="./" ): 

        file = open( folder+"formfactor_1P_pars.jk", "w" ) 

        for line in self.formfactor_pars: 
            file.write( "{0:.20f} {1:.20f}\n".format( line[0], line[1] ) )

        file.close() 

    def get_formfactor_from_attributes( self, Q2 ): 

        mu2_jk = self.formfactor_pars.T[0] 
        A_jk   = self.formfactor_pars.T[1]

        def hypth( var ): 
            return 1.0 / ( var / mu2_jk + 1.0 ) + A_jk * Q2

        Q2 = np.array( Q2 ) 
        if Q2.ndim: 
            return hypth( Q2 ) 
        else: 
            return np.array( [ hypth(_) for _ in Q2 ] )   

    def get_formfactor_from_file( self, Q2, path ): 

        pars_jk = [] 
        for line in open( path, "r" ): 
            pars_jk.append( np.fromstring( line, dtype=float, sep=" " ) ) 
        pars_jk = np.array( pars_jk ).T 

        mu2_jk = pars_jk[0] 
        A_jk   = pars_jk[1] 

        def hypth( var ): 
            return 1.0 / ( var / mu2_jk + 1.0 ) + A_jk * Q2 
        
        Q2 = np.array( Q2 ) 
        if Q2.ndim: 
            return hypth( Q2 ) 
        else: 
            return np.array( [ hypth(_) for _ in Q2 ] )   
        

class two_2pt( object ): 

    def __init__( self, NLSM, TNOT=0, SIZE=3, BOOST=0, SKIP=0, TMAX=24, COPIES=1000 ): 

        self.NLSM   = NLSM 
        self.TNOT   = TNOT 
        self.SIZE   = SIZE 
        self.BOOST  = BOOST 
        self.SKIP   = SKIP 
        self.TMAX   = TMAX 
        self.COPIES = COPIES

        self.TIME = NLSM.TIME 
        self.LEN  = NLSM.LEN  
        self.UNIT = NLSM.UNIT 

        self.corr_matrix = [] 
        self.pcorrs      = [] 
        self.interps     = [] 
        self.spectrum    = [] 
        self.Chi2dof     = [] 

        ## self.interps = [  ]

    def read_corrs( self, paths: list, im_paths=False, VISUAL=False )->None: 

        all_correlators = [] 

        for i in range( self.SIZE ): 
            aux_i = [] 
            for j in range( self.SIZE ): 

                aux_ij = file_to_jk( paths[i][j] ) 
                if im_paths: aux_ij += 1.0j*file_to_jk( im_paths[i][j] ) 

                aux_t  = [] 
                for t in range( self.TMAX ): 
                    aux_t.append( aux_ij[t] )
                    
                aux_i.append( aux_t ) 

            all_correlators.append( aux_i ) 

        symm_corrs = [] 
        for i in range( self.SIZE ): 
            aux_i = [] 
            for j in range( self.SIZE ):
                aux_ij = [] 
                for t in range( self.TMAX ): 
                    aux_ij.append( 0.5*( all_correlators[i][j][t] + all_correlators[j][i][t].conjugate() ) )
                aux_i.append( aux_ij ) 
            symm_corrs.append( aux_i ) 

        if VISUAL: self.corr_visual = symm_corrs

        matrices_jk = [] 
        for k in range( len( symm_corrs[0][0][0] ) ): 
            kth_copy = [] 
            for t in range( self.TMAX ): 
                kth_copy_at_t = [] 
                for i in range( self.SIZE ): 
                    row = [] 
                    for j in range( self.SIZE ): 
                        row.append( symm_corrs[i][j][t][k] ) 
                    kth_copy_at_t.append( row )
                kth_copy.append( np.matrix( kth_copy_at_t ) ) 
            matrices_jk.append( kth_copy ) 
        self.corr_matrix = matrices_jk

    def define_metric( self, C0: np.matrix )->tuple:  

        vals, vecs    = np.linalg.eigh( C0 ) 
        Sigma         = np.matrix( np.diag( vals ) ) 
        InvSqrt_Sigma = np.linalg.inv( lin.sqrtm( Sigma ) ) 

        right = vecs @ InvSqrt_Sigma 
        left  = InvSqrt_Sigma @ ( vecs.T ) 

        return right, left 
    
    def normalize( self, interps: np.matrix )->np.matrix:

        normalized_interps = [] 

        for n in range( self.SIZE ): 

            vec = interps.T[n].T

            vec  /= np.sqrt( ( vec.H @ vec )[0,0] ) 
            index = np.unravel_index( np.argmax( np.abs( vec ) ), np.shape( vec ) )
            phase = np.angle( vec[index] )  ## CHECK CONVENTIONS 
            vec  *= np.exp( -1.0j * phase ) 

            normalized_interps.append( vec ) 

        return np.concatenate( normalized_interps, axis=1 )


    def GEVP_to_EVP( self, C0: np.matrix, Ct: np.matrix )->tuple: 

        right, left = self.define_metric( C0 ) 

        EVP           = left @ ( Ct @ right ) 
        gvals, next_U = np.linalg.eigh( EVP ) 

        interpolators_next_not_norm = right @ next_U 
        interpolators_next          = self.normalize( interpolators_next_not_norm ) 

        return gvals, interpolators_next 
    
    def find_order( self, Ct: np.matrix, interps_next: np.matrix, interps_prev: np.matrix )->list:

        test_orth = np.abs( interps_prev.H @ Ct @ interps_next ) 
        reorder   = [] 

        while test_orth.any(): 

            row, col = np.unravel_index( np.argmax( np.abs( test_orth ) ), np.shape( test_orth ) ) 
            test_orth[ row, : ] = 0 
            test_orth[ :, col ] = 0 
            reorder.append( ( row, col ) ) 

        return reorder 
    
    def sort_vecs( self, interps_next: np.matrix, gvals: np.ndarray, order: list )->tuple:

        new_pcorrs  = np.zeros( self.SIZE )+0.0j 
        new_interps = np.matrix( np.zeros( (self.SIZE, self.SIZE) )+0.0j ) 
        for prev, next in order: 
            new_pcorrs[  prev ] = gvals[ next ] 
            new_interps[ prev ] = interps_next.T[ next ] 

        return new_pcorrs, new_interps.T 

    def solve_GEVP( self )->None: 

        for Ct_list in self.corr_matrix: 

            C0 = Ct_list[ self.TNOT ] 

            interps_prev = np.matrix( [] ) 

            aux_for_interps = [ np.matrix( np.zeros((self.SIZE,self.SIZE))+0.0j ) for _ in range( self.TMAX ) ]
            aux_for_pcorrs  = [ np.zeros( self.SIZE )+0.0j for _ in range(self.TMAX) ] 

            for t in range( len( Ct_list ) ): 

                if t == self.TNOT: continue 

                Ct = Ct_list[t] 

                gvals, interps_next = self.GEVP_to_EVP( C0, Ct ) 

                if interps_prev.any(): 
                    reorder = self.find_order( Ct, interps_next, interps_prev ) 
                    sorted_gvals, sorted_interps = self.sort_vecs( interps_next, gvals, reorder ) 

                    interps_prev = sorted_interps 
                    aux_for_interps[ t ] = interps_prev 
                    aux_for_pcorrs[  t ] = sorted_gvals 

                else:
                    interps_prev = interps_next
                    aux_for_interps[ t ] = interps_next 
                    aux_for_pcorrs[  t ] = gvals

            self.interps.append( aux_for_interps )
            self.pcorrs.append(  aux_for_pcorrs  )

            #for n in range( self.SIZE ): 
            #    self.pcorrs[  n ].append( aux_for_pcorrs[  n ] ) 
            #    self.interps[ n ].append( aux_for_interps[ n ] ) 

        #self.pcorrs  = np.array( self.pcorrs ) 
        #self.interps = np.array( self.interps ) 

    def save_corr_data( self, path="./" ):

        Path( path+"New_PCorrs" ).mkdir( parents=True, exist_ok=True ) 
        Path( path+"New_PCorrs/TNOT_is_{}".format( self.TNOT ) ).mkdir( parents=True, exist_ok=True )

        folder = path+"./New_PCorrs/TNOT_is_{}".format( self.TNOT ) 

        for n in range( self.SIZE ): 
            ### GENERALIZE FOR OTHER CHANNELS 
            re_file = open( folder + "/2P_I1_re_PCorrs_E{0}_P{1}.dat".format( n, self.BOOST ), "w" ) 
            im_file = open( folder + "/2P_I1_im_PCorrs_E{0}_P{1}.dat".format( n, self.BOOST ), "w" ) 
            for data in  self.pcorrs: 
                line = [ data[t][n] for t in range( self.TMAX ) ] 
                for val in line:
                    re_file.write( "{0:.24f} ".format( val.real ) )
                    im_file.write( "{0:.24f} ".format( val.imag ) ) 
                re_file.write( "\n" )
                im_file.write( "\n" ) 
            re_file.close() 
            im_file.close() 

        Path( path+"New_Interps" ).mkdir( parents=True, exist_ok=True ) 
        Path( path+"New_Interps/TNOT_is_{}".format( self.TNOT ) ).mkdir( parents=True, exist_ok=True )

        folder = path+"New_Interps/TNOT_is_{}".format( self.TNOT ) 

        for n in range( self.SIZE ): 
            re_file = open( folder + "/2P_I1_re_PCorrs_E{0}_P{1}.dat".format( n, self.BOOST ), "w" ) 
            im_file = open( folder + "/2P_I1_im_PCorrs_E{0}_P{1}.dat".format( n, self.BOOST ), "w" ) 
            for data in self.interps: 
                line = [ data[t].T[n] for t in range( self.TMAX ) ] 
                for vec in line: 
                    counter = 0
                    for val in [ vec[0,_] for _ in range( self.SIZE ) ]: 
                        re_file.write( "{0:.24f}".format(val.real) )
                        im_file.write( "{0:.24f}".format(val.imag) )
                        if counter < self.SIZE-1: 
                            re_file.write( ",".format(val.real) )
                            im_file.write( ",".format(val.imag) )
                        counter+=1
                    re_file.write(" ")
                    im_file.write(" ")
                re_file.write("\n")
                im_file.write("\n") 
            re_file.close() 
            im_file.close() 

    def make_fit( self, ti: int, tf: int, STATE: int, GUESS=0.5, CHI2dof=False )->None: 

        if ti == self.TNOT: 
            print( "ERROR: ti==TNOT\n" )
            return None
        
        if self.TNOT > 0: 
            pointer = STATE 
        else: 
            pointer = self.SIZE-1-STATE 

        if ti > self.TNOT:
            t_fit  = np.arange( ti, tf+1 ) 
        else: 
            t_fit  = np.array( [ _ for _ in range(ti,self.TNOT) ]
                               +[ _ for _ in range(self.TNOT+1,tf+1) ] ) 
        
        dof = len( t_fit ) - 1 

        def hypth( E, t ):
            return np.exp( - E * ( t - self.TNOT ) ) 
        
        #jk_ensemble = np.array( [ [ self.pcorrs[k][t][pointer] for k in range(self.COPIES) ] for t in t_fit ] ) 
        ## FORCING THE REAL PART - CHECK!!!
        jk_ensemble = np.array( [ [ self.pcorrs[k][t][pointer].real for k in range(self.COPIES) ] for t in t_fit ] ) 

        cov_matrix = jk.jk_cov_matrix( jk_ensemble ) 
        InvCov     = np.linalg.inv( cov_matrix ) 

        E_jk     = [] 
        Chi2norm = []

        for data in jk_ensemble.T:

            def Chi2( E ):
                fit = hypth( E, t_fit )
                diffs = np.matrix( data - fit ).T 
                return ( diffs.T @ InvCov @ diffs )[0,0]
            
            Chi2_fit = Minuit( Chi2, E=GUESS ) 

            Chi2_fit.errordef = Minuit.LEAST_SQUARES 
            Chi2_fit.strategy = 2 

            Chi2_fit.errors["E"] = 0.000001 

            Chi2_fit.migrad() 

            E_jk.append( Chi2_fit.values[0] ) 

            Chi2norm.append( Chi2_fit.fval / dof ) 

        E_jk     = np.array( E_jk ) 
        Chi2norm = np.array( Chi2norm ) 
        self.Chi2dof.append( Chi2norm )

        if CHI2dof:
            return E_jk, Chi2norm 
        
        else: 
            return E_jk 
        
    def fit_spectrum( self, fit_ranges, GUESS=False ): 

        if not GUESS: GUESS = [0.5]*self.SIZE 

        spectrum = [] 

        for STATE, fit_range in enumerate( fit_ranges ): 

            if self.TNOT > 0: 
                LABEL = STATE 
            else: 
                LABEL = self.SIZE-1-STATE  

            ti, tf = fit_range[0], fit_range[1] 
            spectrum.append( self.make_fit( ti, tf, LABEL, GUESS=GUESS[STATE] ) )

        self.spectrum = np.array( spectrum ) 

    def make_fit_dexp( self, ti: int, tf: int, STATE: int, GUESS=[0.5,1.0,0.01], CHI2dof=False )->None: 

        if ti == self.TNOT: 
            print( "ERROR: ti==TNOT\n" )
            return None
        
        if self.TNOT > 0: 
            pointer = STATE 
        else: 
            pointer = self.SIZE-1-STATE 

        if ti > self.TNOT:
            t_fit  = np.arange( ti, tf+1 ) 
        else: 
            t_fit  = np.array( [ _ for _ in range(ti,self.TNOT) ]
                               +[ _ for _ in range(self.TNOT+1,tf+1) ] ) 
        
        dof = len( t_fit ) - 3 

        def hypth( pars, t ):
            E, A, U = pars[0], pars[1], pars[2]
            return (1.0-A) * np.exp( - E * ( t - self.TNOT ) ) + A * np.exp( - U * ( t - self.TNOT ) )
        
        #jk_ensemble = np.array( [ [ self.pcorrs[k][t][pointer] for k in range(self.COPIES) ] for t in t_fit ] ) 
        ## FORCING THE REAL PART - CHECK!!!
        jk_ensemble = np.array( [ [ self.pcorrs[k][t][pointer].real for k in range(self.COPIES) ] for t in t_fit ] ) 

        cov_matrix = jk.jk_cov_matrix( jk_ensemble ) 
        InvCov     = np.linalg.inv( cov_matrix ) 

        E_jk, A_jk, U_jk = [], [], [] 
        Chi2norm         = []

        for data in jk_ensemble.T:

            def Chi2( E, A, U ):
                fit = hypth( [E,A,U], t_fit )
                diffs = np.matrix( data - fit ).T 
                return ( diffs.T @ InvCov @ diffs )[0,0]
            
            Chi2_fit = Minuit( Chi2, E=GUESS[0], A=GUESS[1], U=GUESS[2] ) 

            Chi2_fit.errordef = Minuit.LEAST_SQUARES 
            Chi2_fit.strategy = 2 

            Chi2_fit.errors["E"] = 0.000001 
            Chi2_fit.errors["A"] = 0.000001 
            Chi2_fit.errors["U"] = 0.000001

            Chi2_fit.limits["A"] = ( 0.0, 0.1 ) 
            Chi2_fit.limits["U"] = ( 3.0*self.NLSM.MASS, 5.0*self.NLSM.MASS )

            Chi2_fit.migrad() 

            E_jk.append( Chi2_fit.values[0] ) 
            A_jk.append( Chi2_fit.values[1] ) 
            U_jk.append( Chi2_fit.values[2] ) 

            Chi2norm.append( Chi2_fit.fval / dof ) 

        E_jk     = np.array( E_jk ) 
        Chi2norm = np.array( Chi2norm ) 

        if CHI2dof:
            return E_jk, A_jk, U_jk, Chi2norm 
        
        else: 
            return E_jk, A_jk, U_jk 
        
class two_3pt( object ):

    def __init__( self, NLSM, TCAP=12 ):

        self.NLSM = NLSM 
        self.UNIT = NLSM.UNIT

        self.TCAP = TCAP 

        self.C2pt_labels = [] 
        self.C2pt_data   = [] 
        self.C2pt_pars   = [] 

        self.C3pt_labels  = [] 
        self.re_C3pt_data = [] 
        self.im_C3pt_data = [] 
        self.phases_data  = []
        self.Q2_vals      = [] 
        self.sf_vals      = [] 
        self.si_vals      = [] 

        self.ratios = [] 
        self.re_ratios = [] 
        self.im_ratios = []
        self.ratios_from_data = [] 
        #self.ratio_fits_in_tc = [] 

        self.state_marker = [ "o", "s", "^" ]
        self.boost_color  = [ "steelblue", "orchid", "goldenrod", "firebrick" ]

    def fit_C2pt_interps( self, STATE: int, BOOST: int, re_path: str, ti: int, tf: int, im_path=False, GUESS=[50000,0.2] )->None: 

        self.C2pt_labels.append( (STATE,BOOST) ) 

        def C2pt_hypth( pars, t ): 
            A, E = pars[0], pars[1] 
            return A * np.exp( - E * t )  
        
        C2pt_jk = file_to_jk( re_path ).real
        
        #### ADD ROUTINE TO TEST CONSISTENCY OF IMAGINRY PART WITH ZERO 
        if im_path: im_C2pt_jk = file_to_jk( im_path ).real

        self.C2pt_data.append( C2pt_jk ) 

        t_fit = np.arange( ti, tf+1, 1 )  

        ensemble_jk = C2pt_jk[ ti : tf+1 ] 

        cov_matrix = jk.jk_cov_matrix( ensemble_jk ) 
        InvCov     = np.matrix( np.linalg.inv( cov_matrix ) ) 

        A_jk, E_jk = [], [] 

        for data in ensemble_jk.T: 

            def Chi2( A, E ): 
                fit   = C2pt_hypth( [A,E], t_fit ) 
                diffs = np.matrix( data - fit ).T 
                return ( diffs.T @ InvCov @ diffs )[0,0] 
            
            Chi2_fit = Minuit( Chi2, A=GUESS[0], E=GUESS[1] ) 

            Chi2_fit.errordef = Minuit.LEAST_SQUARES 
            Chi2_fit.strategy = 2 
            Chi2_fit.errors["A"] = 100
            Chi2_fit.errors["E"] = 0.0005 
            Chi2_fit.migrad() 

            A_jk.append( Chi2_fit.values[0] )
            E_jk.append( Chi2_fit.values[1] ) 

        self.C2pt_pars.append( np.array( [ A_jk, E_jk ] ) ) 

        ## CURRENT IMPLEMENTATION ALLOWS FOR REPETITION OF C2pt, FIX LATER

    def C2pt_from_pars( self, STATE: int, BOOST: int, t: np.ndarray )->np.ndarray: 

        pointer = self.C2pt_labels.index( ( STATE, BOOST ) ) 

        pars_jk = self.C2pt_pars[ pointer ] 

        A_jk = pars_jk[0] 
        E_jk = pars_jk[1]

        if t.ndim: 
            return np.array( [ A_jk * np.exp(-E_jk*val) for val in t ] ) 
        
        else:
            return A_jk * np.exp( - E_jk * t ) 
        
    def read_corrs( self, OUT: tuple, IN: tuple, re_path: str, im_path: str )->None: 

        label = (OUT,IN) 

        re_C3pt_jk = file_to_jk( re_path ).real  
        im_C3pt_jk = file_to_jk( im_path ).real 

        self.C3pt_labels.append( label ) 
        self.re_C3pt_data.append( re_C3pt_jk ) 
        self.im_C3pt_data.append( im_C3pt_jk ) 
        self.phases_data.append( np.zeros( np.shape( re_C3pt_jk ) ) ) 

    def compute_ratio_from_pars( self, label:tuple, factor=1.0 )->None: 

        TCAP        = self.TCAP
        pointer     = self.C3pt_labels.index( label ) 
        STATE_OUT, BOOST_OUT = label[0][0], label[0][1] 
        STATE_IN,  BOOST_IN  = label[1][0], label[1][1] 

        OUT_index = self.C2pt_labels.index( ( STATE_OUT, BOOST_OUT ) ) 
        IN_index  = self.C2pt_labels.index( ( STATE_IN,  BOOST_IN  ) ) 

        OUT_pars  = self.C2pt_pars[ OUT_index ]
        IN_pars   = self.C2pt_pars[ IN_index ] 

        OUT_E_tot, OUT_P_tot = OUT_pars[1], BOOST_OUT * self.UNIT 
        IN_E_tot,  IN_P_tot  = IN_pars[1],  BOOST_IN  * self.UNIT 

        OUT_s = np.square( OUT_E_tot ) - np.square( OUT_P_tot )
        IN_s  = np.square( IN_E_tot ) - np.square( IN_P_tot ) 

        self.sf_vals.append( OUT_s )
        self.si_vals.append( IN_s ) 

        Q2 = np.square( OUT_P_tot - IN_P_tot ) - np.square( OUT_E_tot - IN_E_tot ) 

        self.Q2_vals.append( Q2 ) 

        tc_range = np.arange( -TCAP+1, TCAP, 1 ) 

        re_data_jk = self.re_C3pt_data[ pointer ] 
        im_data_jk = self.im_C3pt_data[ pointer ] 

        abs_C3pt_jk = np.sqrt( np.square(re_data_jk).real + np.square(im_data_jk).real ) 

        abs_ratio_jk = [] 
        re_ratio_jk  = [] 
        im_ratio_jk  = [] 

        for tc in tc_range: 

            pointer_tc = tc+TCAP-1 

            abs_C3pt_jk_at_tc = abs_C3pt_jk[ pointer_tc ] 

            C2pt_OUT_jk_at_tc = self.C2pt_from_pars( STATE_OUT, BOOST_OUT, 2*(TCAP-tc) ) 
            C2pt_IN_jk_at_tc  = self.C2pt_from_pars( STATE_IN,  BOOST_IN,  2*(TCAP+tc) )  

            if label[0] == label[1]: 
                C2pt_OUT_jk_at_tc = self.C2pt_from_pars( STATE_OUT, BOOST_OUT, 2*np.array(TCAP) ) 
                C2pt_IN_jk_at_tc  = self.C2pt_from_pars( STATE_IN,  BOOST_IN,  2*np.array(TCAP) ) 


            ratio_jk_at_tc = factor * abs_C3pt_jk_at_tc / np.sqrt( C2pt_OUT_jk_at_tc * C2pt_IN_jk_at_tc ) 

            re_ratio_jk_at_tc = factor * re_data_jk[pointer_tc] / np.sqrt( C2pt_OUT_jk_at_tc * C2pt_IN_jk_at_tc ) 

            im_ratio_jk_at_tc = factor * im_data_jk[pointer_tc] / np.sqrt( C2pt_OUT_jk_at_tc * C2pt_IN_jk_at_tc ) 

            abs_ratio_jk.append( ratio_jk_at_tc ) 
            re_ratio_jk.append( re_ratio_jk_at_tc )
            im_ratio_jk.append( im_ratio_jk_at_tc )

        self.ratios.append( np.array( abs_ratio_jk ) ) 
        self.re_ratios.append( np.array( re_ratio_jk ) )
        self.im_ratios.append( np.array( im_ratio_jk ) )

    def compute_ratio_from_data( self, label:tuple, JK_COPIES=1000, factor=1.0, TMAX=41 )->None: 

        TCAP    = self.TCAP 
        pointer = self.C3pt_labels.index( label ) 
        STATE_OUT, BOOST_OUT = label[0][0], label[0][1] 
        STATE_IN,  BOOST_IN  = label[1][0], label[1][1] 

        pointer_OUT = self.C2pt_labels.index( label[0] ) 
        pointer_IN  = self.C2pt_labels.index( label[1] ) 

        tc_range = np.arange( -TCAP+1, TCAP, 1 ) 

        re_data_jk = self.re_C3pt_data[ pointer ] 
        im_data_jk = self.im_C3pt_data[ pointer ] 

        abs_C3pt_jk = np.sqrt( np.square(re_data_jk).real + np.square(im_data_jk).real ) 

        abs_ratio_jk = [] 

        for tc in tc_range: 

            if abs(2*(TCAP-tc))>TMAX or abs(2*(TCAP+tc))>TMAX: continue 

            pointer_tc = tc+TCAP-1 

            abs_C3pt_jk_at_tc = abs_C3pt_jk[ pointer_tc ] 

            C2pt_OUT_jk_at_tc = np.abs( self.C2pt_data[pointer_OUT][ 2*(TCAP-tc) ] )
            C2pt_IN_jk_at_tc  = np.abs( self.C2pt_data[pointer_IN ][ 2*(TCAP+tc) ] )

            if label[0] == label[1]:
                C2pt_OUT_jk_at_tc = np.abs( self.C2pt_data[pointer_OUT][ 2*TCAP ] )
                C2pt_IN_jk_at_tc  = np.abs( self.C2pt_data[pointer_IN ][ 2*TCAP ] )

            ratio_jk_at_tc = factor * abs_C3pt_jk_at_tc / np.sqrt( C2pt_OUT_jk_at_tc * C2pt_IN_jk_at_tc )

            abs_ratio_jk.append( ratio_jk_at_tc ) 

        self.ratios_from_data.append( np.array( abs_ratio_jk ) ) 

    def fit_ratio_in_tc( self, label: tuple, ti: int, tf: int, GUESS=0.85, CHI2dof=False, EXCHANGE=False ): 

        ratio_index = self.C3pt_labels.index( label ) 

        tc_range = np.arange( ti, tf+1, 1 ) 

        ni, nf = ti+self.TCAP-1, tf+self.TCAP-1 

        ensemble_jk = np.array( self.ratios[ratio_index][ni:nf+1] ) 
        if EXCHANGE: 
            label_EX = ( label[1], label[0] ) 
            ratio_EX_index = self.C3pt_labels.index( label_EX ) 
            ensemble_jk = 0.5 * ( np.array( self.ratios[ratio_index][ni:nf+1] ) + np.array( self.ratios[ratio_EX_index][ni:nf+1] ) ) 
        cov_matrix  = jk.jk_cov_matrix( ensemble_jk ) 
        InvCov      = np.matrix( np.linalg.inv( cov_matrix ) ) 

        def ratio_hypth( C, t ): 
            return C * np.ones( len( t ) ) 
        
        ratio_fit_jk = [] 
        CHI2dof_jk   = [] 

        for data in ensemble_jk.T: 

            def Chi2( C ): 
                fit   = ratio_hypth( C, tc_range ) 
                diffs = np.matrix( data - fit ).T 
                return ( diffs.T @ InvCov @ diffs )[0,0] 
            
            Chi2_fit = Minuit( Chi2, C=GUESS )  

            Chi2_fit.errordef = Minuit.LEAST_SQUARES 
            Chi2_fit.strategy = 2 
            Chi2_fit.errors["C"] = 0.001 

            Chi2_fit.migrad() 

            ratio_fit_jk.append( Chi2_fit.values[0] ) 

            if CHI2dof: 
                CHI2dof_jk.append( Chi2_fit.fval / (len(ensemble_jk)-1.0) )

        if not CHI2dof:
            return np.array( ratio_fit_jk ) 
        else: 
            return np.array( ratio_fit_jk ), np.array( CHI2dof_jk ) 
        
    def compute_3pt_phase( self, label: tuple )->None: 

        pointer = self.C3pt_labels.index( label ) 

        re_C3pt_jk = self.re_C3pt_data[ pointer ].real 
        im_C3pt_jk = self.im_C3pt_data[ pointer ].real  

        sign_re_jk = np.sign( re_C3pt_jk ) 
        sign_im_jk = np.sign( im_C3pt_jk )  

        pre_phase_jk = np.arctan( im_C3pt_jk / re_C3pt_jk ).real 

        phase_jk = [] 

        for nt in range( 2*self.TCAP-1 ): 

            phase_aux = [] 

            #print()
            #print( -self.TCAP+1+nt )
            #print( np.mean( im_C3pt_jk[nt]/re_C3pt_jk[nt] ) )
            #print()

            for nu, phase in enumerate( pre_phase_jk[nt] ): 
    
                if sign_im_jk[nt][nu] >= 0: 
                    if sign_re_jk[nt][nu] >= 0: 
                        ## 1st Quadrant 
                        phase_aux.append( (phase)%(2.0*np.pi) ) 
                    else: 
                        ## 2nd Quadrant 
                        ##phase_aux.append( (np.pi - phase)%(2.0*np.pi) ) ## CHECK <09/03/25>
                        phase_aux.append( (np.pi + phase)%(2.0*np.pi) ) 
                else: 
                    if sign_re_jk[nt][nu] <0 :
                        ## 3rd Quadrant 
                        phase_aux.append( (np.pi + phase)%(2.0*np.pi) ) 
                    else: 
                        ## 4th Quadrant 
                        ##phase_aux.append( (2.0*np.pi - phase)%(2.0*np.pi) )  ## CHECK <09/03/25>
                        phase_aux.append( (2.0*np.pi + phase)%(2.0*np.pi) ) 

            phase_jk.append( phase_aux )

        self.phases_data[ pointer ] = np.array( phase_jk )/np.pi

    def fit_re_ratios_in_tc( self, label: tuple, ti: int, tf: int, GUESS=0.85, CHI2dof=False, EXCHANGE=False ): 

        re_ratios_index = self.C3pt_labels.index( label ) 

        tc_range = np.arange( ti, tf+1, 1 ) 

        ni, nf = ti+self.TCAP-1, tf+self.TCAP-1 

        ensemble_jk = np.array( self.re_ratios[re_ratios_index][ni:nf+1] )
        if EXCHANGE: 
            label_EX = ( label[1], label[0] ) 
            re_ratios_EX_index = self.C3pt_labels.index( label_EX ) 
            ensemble_jk = 0.5 * ( np.array( self.re_ratios[re_ratios_index][ni:nf+1] ) + np.array( self.re_ratios[re_ratios_EX_index][ni:nf+1] ) ) 
        cov_matrix  = jk.jk_cov_matrix( ensemble_jk ) 
        InvCov      = np.matrix( np.linalg.inv( cov_matrix ) ) 

        def hypth( C, t ): 
            return C * np.ones( len( t ) ) 
        
        fit_jk       = [] 
        CHI2dof_jk   = [] 

        for data in ensemble_jk.T: 

            def Chi2( C ): 
                fit   = hypth( C, tc_range ) 
                diffs = np.matrix( data - fit ).T 
                return ( diffs.T @ InvCov @ diffs )[0,0] 
            
            Chi2_fit = Minuit( Chi2, C=GUESS )  

            Chi2_fit.errordef = Minuit.LEAST_SQUARES 
            Chi2_fit.strategy = 2 
            Chi2_fit.errors["C"] = 0.001 

            Chi2_fit.migrad() 

            fit_jk.append( Chi2_fit.values[0] ) 

            if CHI2dof: 
                CHI2dof_jk.append( Chi2_fit.fval / (len(ensemble_jk)-1.0) )

        if not CHI2dof:
            return np.array( fit_jk ) 
        else: 
            return np.array( fit_jk ), np.array( CHI2dof_jk ) 
        
    def fit_im_ratios_in_tc( self, label: tuple, ti: int, tf: int, GUESS=0.85, CHI2dof=False, EXCHANGE=False ): 

        im_ratios_index = self.C3pt_labels.index( label ) 

        tc_range = np.arange( ti, tf+1, 1 ) 

        ni, nf = ti+self.TCAP-1, tf+self.TCAP-1 

        ensemble_jk = np.array( self.im_ratios[im_ratios_index][ni:nf+1] )
        if EXCHANGE: 
            label_EX = ( label[1], label[0] ) 
            im_ratios_EX_index = self.C3pt_labels.index( label_EX ) 
            ensemble_jk = 0.5 * ( np.array( self.im_ratios[im_ratios_index][ni:nf+1] ) - np.array( self.im_ratios[im_ratios_EX_index][ni:nf+1] ) ) 
        cov_matrix  = jk.jk_cov_matrix( ensemble_jk ) 
        InvCov      = np.matrix( np.linalg.inv( cov_matrix ) ) 

        def hypth( C, t ): 
            return C * np.ones( len( t ) ) 
        
        fit_jk       = [] 
        CHI2dof_jk   = [] 

        for data in ensemble_jk.T: 

            def Chi2( C ): 
                fit   = hypth( C, tc_range ) 
                diffs = np.matrix( data - fit ).T 
                return ( diffs.T @ InvCov @ diffs )[0,0] 
            
            Chi2_fit = Minuit( Chi2, C=GUESS )  

            Chi2_fit.errordef = Minuit.LEAST_SQUARES 
            Chi2_fit.strategy = 2 
            Chi2_fit.errors["C"] = 0.001 

            Chi2_fit.migrad() 

            fit_jk.append( Chi2_fit.values[0] ) 

            if CHI2dof: 
                CHI2dof_jk.append( Chi2_fit.fval / (len(ensemble_jk)-1.0) )

        if not CHI2dof:
            return np.array( fit_jk ) 
        else: 
            return np.array( fit_jk ), np.array( CHI2dof_jk ) 

        
        






    

            



    

    





                    


    
    
        