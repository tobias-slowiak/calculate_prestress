from netgen.geom2d import unit_square
from ngsolve import *
from math import pi
import scipy.linalg
from numpy import random
from ngsolve.webgui import Draw
from IPython.display import clear_output
import numpy as np 
from netgen.geom2d import *
from ngsolve.krylovspace import CGSolver
import matplotlib.pyplot as plt
from datetime import datetime
import ast
from itertools import product

### Installation of the required packages
# pip install --upgrade numpy
# pip install --upgrade scipy
# pip install --upgrade matplotlib
# pip install --upgrade ngsolve
# pip install --upgrade webgui_jupyter_widgets



#GENERATE THE MESH
def dist_circ(xmid, ymid, x, y):
    return sqrt((x-xmid)**2 + (y-ymid)**2)

def points_on_circle(p, base_radius, del_radius, h, xmid, ymid):
    radius = base_radius + del_radius
    points = []
    del_angle = h / radius
    #this might cause 2 very close points which is not good for the mesh angles so here i find the the closest whole numbered Teiler to the circumference
    n = 2 * np.pi / del_angle
    n = np.round(n)
    del_angle = 2 * np.pi / n    
    for angle in np.arange(-pi/2, 3*pi/2, del_angle):
        x_pos = xmid + radius * np.cos(angle)
        y_pos = ymid + radius * np.sin(angle)
        if x_pos > 0 and x_pos < 1e-3:
            if y_pos > 0 and y_pos < 1e-3:
                points.append((x_pos, y_pos))
    return points

def points_on_parabola(p, dist_from_middle, rl):
    h = p["h_elements_electrode"]
    points = []
    for y_pos_mid in np.arange(0, p["Lside"], h): # split the y axis into 1e-3/h parts, then shift up or down to get element boundaries perp to electrode
        el_bd_derivative = 2 * p["a"] * y_pos_mid + p["b"] # derivative of the el boundary to shift the y_pos of the element vertices
        y_pos = y_pos_mid + el_bd_derivative * dist_from_middle
        x_pos = p["a"] * y_pos**2 + p["b"] * y_pos + p["c"]
        x_pos += dist_from_middle
        if rl == "r":
            x_pos = p["Lside"] - x_pos
        if x_pos > 0 and x_pos < 1e-3: # checking if point in domain - should be the case, check not really necessary
            if y_pos > 0 and y_pos < 1e-3:
                points.append((x_pos,y_pos))
    return points

def points_on_electrode(p, dist_from_middle, lr):
    if p["el_form"] == "circle":
        if lr == "l":
            xmid, ymid = p["xmidl"], p["ymidl"]
        else:
            xmid, ymid = p["xmidr"], p["ymidr"]
        return points_on_circle(p, p["Rmid"], dist_from_middle, p["h_elements_electrode"], xmid, ymid)
    elif p["el_form"] == "parabola":
        return points_on_parabola(p, dist_from_middle, lr)

def MakeGeometry(p):
    geometry = SplineGeometry()

    #BOUNDARIES
    boundary_points = [(0,0), (p['Lside'],0), (p['Lside'],p['Lside']), (0,p['Lside'])]
    pnums = [geometry.AppendPoint(*point) for point in boundary_points]

    # start-point, end-point, boundary-condition, domain on left side, domain on right side:
    lines = []
    for i in range(3):
        lines.append((pnums[i],pnums[i+1],"fix",1,0))
    lines.append((pnums[3],pnums[0],"fix",1,0))

    #ELECTRODES
    del_w, n_layers, dists_from_middle = 0, 0, []
    if p['n_elements_electrode'] == 0: # normal case
        n_layers = 1
    else: # for convergence analysis this makes it possible to resolve the electrode better
        n_layers = p['n_elements_electrode']
        if n_layers % 2 == 0:# this only works for uneven numbers of elements across electrode, no special reason, just lazyness
            n_layers += 1
    del_w = p["el_width"] / n_layers
    dists_from_middle = [(-del_w / 2 - i * del_w) for i in range(n_layers)] + [(+del_w / 2 + i * del_w) for i in range(n_layers)]
    for dist_from_middle in dists_from_middle:
        for lr in ["l", "r"]:
            BC = ""
            if lr == "l":
                BC += "left_"
            elif lr == "r":
                BC += "right_"
            if dist_from_middle == (-del_w / 2 - np.floor(n_layers / 2) * del_w):
                BC += "inner"
            elif dist_from_middle == (+del_w / 2 + np.floor(n_layers / 2) * del_w):
                BC += "outer"
            pnts = points_on_electrode(p, dist_from_middle, lr)
            pnums = []
            for point in pnts:
                pnums.append(geometry.AppendPoint(*point))
            for i in range(len(pnums)-1):
                lines.append((pnums[i],pnums[i+1],BC,1,1))

    #PERFORATED AREA
    if p["holes"]:
        #CASE: actual perforation
        for x, y in product(np.arange(0, p["Lside"] + p["dist_holes"], p["dist_holes"]), repeat=2):
            if np.sqrt((x-p["Lside"]/2)**2 + (y-p["Lside"]/2)**2) < p["Rperf"]:
                pnts = points_on_circle(p, p["Rholes"], 0, p["Rholes"]/4, x, y)
                pnums = []
                for point in pnts:
                    pnums.append(geometry.AppendPoint(*point))
                for i in range(len(pnums)-1):
                    lines.append((pnums[i],pnums[i+1],"hole_bnd",0,1))
                lines.append((pnums[-1],pnums[0],"hole_bnd",0,1))
    else:
        #CASE: effective value on perforation
        pnts = points_on_circle(p, p["Rperf"], 0, p["h_elements_perf"], p["Lside"]/2, p["Lside"]/2)
        pnums = []
        for point in pnts:
            pnums.append(geometry.AppendPoint(*point))
        for i in range(len(pnums)-1):
            lines.append((pnums[i],pnums[i+1],"perf_border",1,1))
        lines.append((pnums[-1],pnums[0],"perf_border",1,1))

    #DROP MASSES WITH COFFEE RING
    for added_mass in p["added_masses"]:
        x_coord, y_coord, radius, coffee_ring_width = added_mass["x_coord"], added_mass["y_coord"], added_mass["radius"], added_mass["coffee_ring_width"]
        pnts_dic = {}
        pnts_dic["inner"] = points_on_circle(p, radius - coffee_ring_width, 0, added_mass["h_elements_coffee"], x_coord, y_coord)
        pnts_dic["outer"] = points_on_circle(p, radius, 0, added_mass["h_elements_coffee"], x_coord, y_coord)
        pnums = []
        for point in pnts_dic["outer"]:
            pnums.append(geometry.AppendPoint(*point))
        for i in range(len(pnums)-1):
            lines.append((pnums[i],pnums[i+1],"coffee_ring_outer",1,1))
        lines.append((pnums[-1],pnums[0],"coffee_ring_outer",1,1))
        if coffee_ring_width > 0:
            pnums = []
            for point in pnts_dic["inner"]:
                pnums.append(geometry.AppendPoint(*point))
            for i in range(len(pnums)-1):
                lines.append((pnums[i],pnums[i+1],"coffee_ring_inner",1,1))
            lines.append((pnums[-1],pnums[0],"coffee_ring_inner",1,1))
    
    #add lines to geometry
    for p1,p2,bc,left,right in lines:
        geometry.Append( ["line", p1, p2], bc=bc, leftdomain=left, rightdomain=right)
    return geometry

def generate_mesh(p):
    return Mesh(MakeGeometry(p).GenerateMesh(maxh=p["hmax"]))


#Helper Functions
def str_to_nparray(s):
    return np.array(ast.literal_eval(s))
def str_to_list(s):
    return ast.literal_eval(s)
def interpolate_nans(y):
    y = np.asarray(y, dtype=np.float64)  # Ensure y is a NumPy array of floats
    x = np.arange(len(y))  # Indices of the array
    mask = ~np.isnan(y)  # Mask for valid (non-NaN) values

    return np.interp(x, x[mask], y[mask])


def prestress_estimator(freq11):
    #TODO: recalculate the coefficients when the standard params are final
    coeffs = [5.08285618e-03, 2.15515599e+00, -1.63964169e+06] #coefficients of the quadratic fit of experimental data
    return np.polyval(coeffs, freq11)



#GEOMETRY OF THE ELECTRODES AND PERFORATION FOR THE COEFFICIENT FUNCTIONS
#TODO: check if i get fluctuations of values on the mesh (of sig/rho on electrode)!! maybe also use eps here maybe then also use from p['eps']
#TODO: make nicer to geom_el(p, side) and pass "left" or "right" as side


###OLD VERSION OF DENSTY
def exponent(p, xmid, ymid, R):
    return - 2 * p["k"] * (dist_circ(xmid, ymid, x, y) - R)
def geom_el(p, xmid, ymid, approx = True): #geometry of the electrode
    if approx:
        return (1 / (1 + exp(exponent(p, xmid, ymid, p["Ri"])))) - (1 / (1 + exp(exponent(p, xmid, ymid, p["Ro"]))))
    else:
        return ceil(dist_circ(xmid, ymid, x, y) - p["Ri"]) + ceil(-dist_circ(xmid, ymid, x, y) + p["Ro"]) - 1
def geom_el_both(p, approx = True):
    return geom_el(p, p["xmidl"], p["ymidl"], approx) + geom_el(p, p["xmidr"], p["ymidr"], approx)
"""
#old version with approx
def generate_rho_fct(p):
    return CF(p["hsin"] * p["rhosin"] * (1-(1-p["mpercentage_perf"])*geom_perf(p)) + p["hcr"] * p["rhocr"] * geom_el_both(p) + p["hau"] * p["rhoau"] * geom_el_both(p))
"""


###PARABOLA VERSION OF THE ELECTRODES
#TODO: ask the ngsolve people to implement a < and > operator for x and y
def geom_el_parabola(p, x, y): #2D function that returns 1 on the electrode and 0 elsewhere
    eps = 1e-6 #makes sure that the function values dont miss near vertices
    return ceil(x - (p["a"] * y**2 + p["b"] * y + p["c"] - p["el_width"] / 2 - eps)) * ceil(-x + (p["a"] * y**2 + p["b"] * y + p["c"] + p["el_width"] / 2 + eps))


#TODO: find out what a good value for eps is and then maybe put it in p
def step_fct_el(p, r):
    eps = 1e-6 #makes sure that the function values dont miss near vertices
    left = p["Rmid"] - p["el_width"] / 2 - eps
    right = p["Rmid"] + p["el_width"] / 2 + eps
    return (ceil((r - left)) - ceil((r - right)))
def step_fct_el_2(p, r): # possibility to make the electrode artificially wider
    eps = 1e-6 #makes sure that the function values dont miss near vertices
    left = p["Rmid"] - p["el_width2"] / 2 - eps
    right = p["Rmid"] + p["el_width2"] / 2 + eps
    return (ceil((r - left)) - ceil((r - right)))
def geom_el_circle(p, x, y):
    if p["ultra_wide_electrode"]:
        return step_fct_el_2(p, dist_circ(p["xmidl"], p["ymidl"], x, y))
    else:
        return step_fct_el(p, dist_circ(p["xmidl"], p["ymidl"], x, y))

def geom_el_both(p, x, y):
    if p["el_form"] == "circle":
        return geom_el_circle(p, x, y) + geom_el_circle(p, (-x + 1e-3), y)
    elif p["el_form"] == "parabola":
        return geom_el_parabola(p, x, y) + geom_el_parabola(p, (-x + 1e-3), y)



#TODO: check if i get fluctuations of values on the mesh (of rho)!! maybe also use eps here maybe then also use from p['eps']
def geom_perf(p):
    return ceil(-dist_circ(p["Lside"]/2, p["Lside"]/2, x, y) + p["Rperf"])



#COEFFICIENT FUNCTIONS
def step_fct(left, right, r):
    return (ceil((r - left)) - ceil((r - right)))
#prestress
#TODO: make this look nicer, i think there were problems by splitting in different functions - check
def generate_sig_fct(p):
    base_sig = p["sigsin"] * p["hsin"]
    el_add_sig = p["hcr"] * p["sigcr"] + p["hau"] * p["sigau"]
    principal_sig = base_sig * (1+(p["corr_perf_sig"])*geom_perf(p)) +  el_add_sig * geom_el_both(p, x, y)
    for added_mass in p["added_masses"]:
        coffee_prestress, non_coffee_prestress = added_mass["coffee_prestress"] , added_mass["non_coffee_prestress"]
        radius, coffee_ring_width = added_mass["radius"], added_mass["coffee_ring_width"]
        x_coord, y_coord = added_mass["x_coord"], added_mass["y_coord"]
        coffee_height, non_coffee_height = added_mass["coffee_height"], added_mass["non_coffee_height"]
        eps = 1e-6
        #TODO: think deeply about what is better: -eps or +eps in the following. I think it drastically changes the total added sig
        step_inner = ceil((radius - coffee_ring_width - eps) - dist_circ(x_coord, y_coord, x, y))
        step_outer = ceil((radius + eps) - dist_circ(x_coord, y_coord, x, y))
        principal_sig += non_coffee_prestress * non_coffee_height * step_inner
        principal_sig += coffee_prestress * coffee_height * (step_outer - step_inner)
    sigfct = CF((( principal_sig ,0),(0,principal_sig )))
    return sigfct

#density
     
def generate_rho_fct(p):
    rhofct = CF(0)
    if len(p["added_masses"]) > 0:
        for added_mass in p["added_masses"]:
            eps = 1e-6
            x_coord, y_coord, mass, radius, coffee_ring_width, coffee_ring_perctg = added_mass["x_coord"], added_mass["y_coord"], added_mass["mass"], added_mass["radius"], added_mass["coffee_ring_width"], added_mass["coffee_ring_perctg"]
            coffee_mass = mass * coffee_ring_perctg
            noncoffee_mass = mass - coffee_mass
            noncoffee_area = np.pi * (radius - coffee_ring_width)**2
            coffee_area = np.pi * radius**2 - noncoffee_area
            noncoffee_density = noncoffee_mass / noncoffee_area
            coffee_density = coffee_mass / coffee_area
            #TODO: think deeply about what is better: -eps or +eps in the following. I think it drastically changes the total added mass
            step_inner = ceil((radius - coffee_ring_width - eps) - dist_circ(x_coord, y_coord, x, y))
            step_outer = ceil((radius + eps) - dist_circ(x_coord, y_coord, x, y))
            rhofct += noncoffee_density * step_inner
            rhofct += coffee_density * (step_outer - step_inner)
    rhofct += p["rhosin"] * p["hsin"] * (1-(1-p["mpercentage_perf"])*geom_perf(p))
    rhofct += p["el_add_rho"] * (geom_el_both(p, x, y))
    return rhofct




#MODE NUMBER DETECTION

def mode_detection(p, gfu, mesh):
    gfnp = np_gridfct(p, gfu, mesh)
    gfnp_normed = gfnp / np.max(gfnp)
    row_sums, col_sums = np.zeros(len(gfnp)), np.zeros(len(gfnp))
    for y_ind in range(len(gfnp)):
         row_sums[y_ind] = np.sum(gfnp_normed[:, y_ind])
         for x_ind in range(len(gfnp)):
              col_sums[x_ind] = np.sum(gfnp_normed[x_ind, :])
    sumss = [row_sums, col_sums]
    n_maxs = [0,0]
    for i in range(2):
        maxs, properties = scipy.signal.find_peaks(sumss[i], prominence=2)
        n_maxs[i] = len(maxs)
    return "(" + str(n_maxs[1]) + "," + str(n_maxs[0]) + ")"

def result_dict(p, mesh, feswave, multigfuwave, eigenvals):
    restult = {}
    modes = ["(1,1)", "(1,2)", "(2,1)","(1,3)", "(3,1)"]
    for i in range(p["num_modes"]):
        gfumode = GridFunction(feswave)
        gfumode.vec.data = multigfuwave.vecs[i].data
        current_mode = mode_detection(p, gfumode, mesh)
        if current_mode in modes:
            restult[current_mode] = [np.sqrt(eigenvals[i]) / (2 * np.pi) * 1e-3, gfumode]
    return restult



#find mode number old version searching for maxima did not work well
# def mode_detection(p, gfu, mesh):
#     gfnp = np_gridfct(p, gfu, mesh)
#     # finding the maxima (extrema, took abs of function values)
#     footprint = np.ones((3,3))
#     maxs = (gfnp == scipy.ndimage.maximum_filter(gfnp, footprint=footprint, mode='constant'))
#     maxs_ind = np.argwhere(maxs)
#     print(maxs_ind)
#     # find all unique x and y positions of extrema
#     maxs_ind_cluster = np.copy(maxs_ind[:,0]) #clustered x indices of extrema
#     maxs_multiplicity = np.ones(len(maxs_ind))
#     eps = np.ceil((p["Lside"] / 10) / p["hmax"]) # maximum index difference of two extrema in "same" x/y-position - right now set to 1 10th of length
#     for ind in range(len(maxs_ind)):
#         for ind2 in range(len(maxs_ind)):
#             if ind != ind2:
#                 # check if ind and ind2 share an x direction
#                 if abs(maxs_ind[ind,0] - maxs_ind[ind2,0]) < eps:
#                     maxs_ind_cluster[ind] = min(maxs_ind_cluster[ind], maxs_ind_cluster[ind2])
#                     # check if ind and ind2 have different y coordinates
#                     # the minimal multiplicity will determine the mode nr in y-direction. this is needed bc the middle maximum in the 3,1 mode sometimes splits in 2.
#                     if abs(maxs_ind[ind,1] - maxs_ind[ind2,1]) > eps:
#                         maxs_multiplicity[ind] += 1
#     maxs_ind_cluster = np.unique(maxs_ind_cluster) # now only contains the unique x-positions of clusters
#     return "(" + str(len(maxs_ind_cluster)) + "," + str(int(min(maxs_multiplicity))) + ")"



#SOLVE THE PDES

#Elasticity Problem


def Stress(p, strain): # TODO : not including the elasticity of the electrodes here, is that ok?
    E_fct = CF(p["Esin"] * (1-(1-p["Epercentage_perf"])*geom_perf(p)))
    mu_fct = CF(E_fct / 2 / (1+p["nu"]))
    lam_fct = CF(E_fct * p["nu"] / (1-p["nu"]*p["nu"]))
    return p["hsin"] * (2*mu_fct*strain + lam_fct*Trace(strain)*Id(2))



def force_CF(p, x, y):
    abs_f = p["hcr"] * p["sigcr"] + p["hau"] * p["sigau"]
    if p["el_form"] == "circle":
        dir_vec_len_l = dist_circ(p["xmidl"], p["ymidl"], x, y)
        dir_vec_len_r = dist_circ(p["xmidr"], p["ymidr"], x, y)
        #components of unit vector in force direction
        x_comp_l = (x-p["xmidl"]) / dir_vec_len_l
        y_comp_l = (y-p["ymidl"]) / dir_vec_len_l
        x_comp_r = (x-p["xmidr"]) / dir_vec_len_r
        y_comp_r = (y-p["ymidr"]) / dir_vec_len_r
        force_l = CF( (abs_f * x_comp_l, abs_f * y_comp_l) )
        force_r = CF( (abs_f * x_comp_r, abs_f * y_comp_r) )
    elif p["el_form"] == "parabola":
        dir_vec_len = sqrt(1 + (2*p["a"]*y+p["b"])**2)
        x_comp_l =  1.0 / dir_vec_len
        y_comp_l = -(2*p["a"]*y+p["b"]) / dir_vec_len
        x_comp_r = -1.0 / dir_vec_len
        y_comp_r = -(2*p["a"]*y+p["b"]) / dir_vec_len
        force_l = CF( (abs_f * x_comp_l, abs_f * y_comp_l) )
        force_r = CF( (abs_f * x_comp_r, abs_f * y_comp_r) )
    force_outer_coffee = CF( (0,0) )
    force_inner_coffee = CF( (0,0) )
    for added_mass in p["added_masses"]:
        abs_f_outer_coffee = added_mass["coffee_height"] * added_mass["coffee_prestress"]
        abs_f_inner_coffee = abs_f_outer_coffee - added_mass["non_coffee_height"] * added_mass["non_coffee_prestress"]
        dir_vec_len = dist_circ(p["Lside"]/2, p["Lside"]/2, x, y)
        #components of unit vector in force direction
        x_comp = (x-p["Lside"]/2) / dir_vec_len
        y_comp = (y-p["Lside"]/2) / dir_vec_len
        force_outer_coffee = CF( (abs_f_outer_coffee * x_comp, abs_f_outer_coffee * y_comp) )
        force_inner_coffee = CF( (abs_f_inner_coffee * x_comp, abs_f_inner_coffee * y_comp) )
    return force_l, force_r, force_outer_coffee, force_inner_coffee

def generate_fes_ela(p, mesh):
    fes = VectorH1(mesh, order=3, dirichlet = "fix")
    u = fes.TrialFunction()
    v = fes.TestFunction()
    gfu = GridFunction(fes)
    force_l, force_r, force_outer_coffee, force_inner_coffee = force_CF(p, x, y)
    f = LinearForm(force_l*v*ds("left_inner") - force_l*v*ds("left_outer") + force_r*v*ds("right_inner") - force_r*v*ds("right_outer")
                   - force_outer_coffee*v*ds("coffee_ring_outer") + force_inner_coffee*v*ds("coffee_ring_inner")).Assemble()
    a = BilinearForm(InnerProduct(Stress(p, Sym(Grad(u))), Sym(Grad(v))).Compile()*dx)
    pre = Preconditioner(a, "bddc")
    a.Assemble()
    return fes, u, v, gfu, f, a, pre

def invert_sys_mat(a, pre, f, gfu, verbose = False):
    inv = CGSolver(a.mat, pre, printrates=verbose, tol=1e-8)
    gfu.vec.data = inv * f.vec
    return gfu

def solve_ela(p, mesh):
    fes, u, v, gfu, f, a, pre = generate_fes_ela(p, mesh)
    gfu = invert_sys_mat(a, pre, f, gfu)
    return gfu

def stress_field(p, gfuela, mesh):
    sigfct = generate_sig_fct(p)
    #TODO: what is the order of the elements here? should it be 2 when it is 3 for fes_ela?
    fesstress = MatrixValued(L2(mesh,order=0), symmetric=True) #the result for elasticity is defined per element, not continuous
    ustress = GridFunction(fesstress)
    ustress.Interpolate(Stress(p, Sym(Grad(gfuela))), fesstress)
    prestress = GridFunction(fesstress)
    prestress.Interpolate(sigfct, fesstress)
    gfstress = ustress + prestress
    return gfstress


# wave problem
def generate_fes_wave(mesh, gfstress, rhofct):
    #use the mean of the normal stresses as a force for the wave equation
    #TODO: is this even right? work out theory!
    #TODO: can I do this more elegantly?
    sigfct_res = 0.5 * (gfstress[0,0] + gfstress[1,1])
    
    feswave = H1(mesh, order=4, dirichlet=[1,2,3,4])
    u = feswave.TrialFunction()
    v = feswave.TestFunction()

    a = BilinearForm(feswave)
    #TODO: am i 100 percent sure this is the right blf
    a += SymbolicBFI(InnerProduct((gfstress * grad(u)),grad(v)))
    pre = Preconditioner(a, "multigrid")

    m = BilinearForm(feswave)
    m += SymbolicBFI(rhofct*u*v)

    a.Assemble()
    m.Assemble()

    return feswave, a, m, pre


def solveAlgEVP(p, feswave, pre, a, m, Verbose = False):
    num = p["num_modes"]
    u = GridFunction(feswave, multidim=num)
    r = u.vec.CreateVector()
    Av = u.vec.CreateVector()
    Mv = u.vec.CreateVector()
    freedofs = feswave.FreeDofs()
    vecs = []
    for i in range(2*num):
        vecs.append (u.vec.CreateVector())
        
    # random initialization of all initial vectors for the iterations
    for v in u.vecs:
        for i in range(len(u.vec)):
           v.data[i] = random.rand() if freedofs[i] else 0
    asmall = Matrix(2*num, 2*num)
    msmall = Matrix(2*num, 2*num)
    lams = num * [1]
    oldlams = num * [1e12]
    while abs(lams[0] - oldlams[0]) + abs(lams[1] - oldlams[1]) + abs(lams[2] - oldlams[2]) + abs(lams[3] - oldlams[3]) > p["convcrit"]:
        oldlams[:] = lams[:]
        for j in range(num):
            vecs[j].data = u.vecs[j]
            r.data = a.mat * vecs[j] - lams[j] * m.mat * vecs[j]
            vecs[num+j].data = pre.mat * r

        for j in range(2*num):
            Av.data = a.mat * vecs[j]
            Mv.data = m.mat * vecs[j]
            for k in range(2*num):
                asmall[j,k] = InnerProduct(Av, vecs[k])
                msmall[j,k] = InnerProduct(Mv, vecs[k])

        ev,evec = scipy.linalg.eigh(a=asmall, b=msmall)
        lams[:] = ev[0:num]
        if Verbose:
            print (i, ":", [lam for lam in lams])

        for j in range(num):
            u.vecs[j][:] = 0.0
            for k in range(2*num):
                u.vecs[j].data += float(evec[k,j]) * vecs[k]
  
    return [lam for lam in lams], u

def solve_wave(p, mesh, gfstress):
    rhofct = generate_rho_fct(p)
    feswave, awave, m, prewave = generate_fes_wave(mesh, gfstress, rhofct)
    eigenvals, multigfuwave = solveAlgEVP(p, feswave, prewave, awave, m)
    return feswave, eigenvals, multigfuwave


#TODO: make generate_fes_ela and generate_fes_wave more unified, same signature etc. and also drop unused variables
def solve(p, freq_to_fit_to = None, mode_to_fit_to = None, ela = True):
    print("solving")
    mesh = generate_mesh(p)
    eigenvals = [0] * p["num_modes"]
    freq_fem = 0
    del_f = 10 * p["convcrit"] # initialize to enter the first iteration
    while abs(del_f) > p["convcrit"]:
        #solve elasticity problem
        gfuela = solve_ela(p, mesh)
        if ela:
            gfstress = stress_field(p, gfuela, mesh)
        else:
            sigfct = generate_sig_fct(p)
            fesstress = MatrixValued(L2(mesh,order=0), symmetric=True) #the result for elasticity is defined per element, not continuous
            prestress = GridFunction(fesstress)
            prestress.Interpolate(sigfct, fesstress)
            gfstress = prestress
        #solve wave problem
        feswave, eigenvals, multigfuwave = solve_wave(p, mesh, gfstress)
        if freq_to_fit_to == None: # accept the result after the first iteration
            del_f = 0
        else:
            for i in range(p["num_modes"]):
                gfmode = GridFunction(feswave)
                gfmode.vec.data = multigfuwave.vecs[i].data
                mode = mode_detection(p, gfmode, mesh)
                if mode == mode_to_fit_to:
                    freq_fem = np.sqrt(eigenvals[i]) / (2 * np.pi)
                    continue
            del_f = freq_to_fit_to - freq_fem
            if abs(del_f) > p["convcrit"]:
                p["sigsin"] += del_f * .5e3 #one MPa results in approx 1 kHz difference in eigenfreq
    return feswave, eigenvals, multigfuwave






# PARAMETERS

def set_p_standard():
    p = {}
    p["Esin"], p["nu"] = 250e9, 0.23 #TODO: are these values right?
    p["sigsin"] = 30.08e6
    p["sigcr"] = 1e9
    p["sigau"] = 40e6
    p["corr_perf_sig"] = 0.00
    p["rhosin"] = 3440  # TODO: should this be 3440 (see wikipedia)
    p["rhocr"] = 7140
    p["rhoau"] = 19320
    p["Lside"] = 1e-3
    #mesh parameters
    p["holes"] = False
    p["Rholes"] = 5e-6
    p["dist_holes"] = 20e-6
    
    p['n_elements_electrode'] = 0 #number of elements across diameter of the electrode
    p['h_elements_electrode'] = 5e-6 #height of the elements along the tangent of the electrodes
    p['h_elements_perf'] = 5e-5
    p["hmax"] = .5e-4
    p["hsin"] = 50e-9
    p["hcr"] = 10e-9
    p["hau"] = 90e-9

    p["el_add_rho"] = p["hcr"] * p["rhocr"] + p["hau"] * p["rhoau"]# TODO: destroy this with el_width2

    ###parameters for the electrodes
    p["el_form"] = "parabola" # either "circle" or "parabola"
    p["el_width"] = 5e-6
    #possibility to make the density of the electrode wider:
    p["ultra_wide_electrode"] = False
    p["el_width2"] = 5e-6 # TODO: destroy everything that i did for the el_width2 afterwards
    #parameters for circle electrode
    p["Ri"] = 0.963e-3
    p["Rmid"] = p["Ri"] + p["el_width"] / 2
    p["Ro"] = p["Ri"] + p["el_width"]
    p["xmidl"] = -0.823e-3
    p["ymidl"] = 0.5e-3
    p["xmidr"] = 0.823e-3 + 1e-3
    p["ymidr"] = 0.5e-3
    #parameters of the parabola for the electrodes
    p["a"] = -560
    p["b"] = 0.56
    p["c"] = 2.5e-6

    #parameters for the perforation
    p["Rperf"] = 0.35e-3
    p["mpercentage_perf"] = 0.62
    p["Epercentage_perf"] = 0.6

    #convergence criterion for the eigenfrequency
    p["convcrit"] = 200

    #number of modes
    p["num_modes"] = 8

    #k for approximation of function which is old
    p['k'] = 10000

    #dropmasses for sample mass distribution
    p["added_masses"] = []#entries dicts with keys (x_coord, y_coord, mass, radius, coffee_ring_width, coffee_ring_perctg, coffee_prestress, non_coffee_prestress, coffee_height, non_coffee_height)
    return p













#PLOTTING AND DATA HANDLING

# approximately transform ngsolve gridfunction in numpy array, somehow didnt find the right ngsolve function
def np_gridfct(p, gfu, mesh):
    N = int(np.ceil(p["Lside"]/p["hmax"]))
    gfnp = np.zeros((N,N))
    x_tmp, y_tmp = np.linspace(0, p["Lside"], N), np.linspace(0, p["Lside"], N)

    if p["holes"]: #weeding out points that might lie in a hole
        for i in range(3):
            for tmp in [x_tmp, y_tmp]:
                for x in tmp:
                    for n in range(int(p["Lside"] / p["dist_holes"])):
                        x_mid = n * p["dist_holes"]
                        if np.abs(x - x_mid) < p["Rholes"] / 2:
                            tmp[tmp == x] += p["Rholes"] 

    x_coords, y_coords = np.meshgrid(x_tmp, y_tmp)
    for x_ind in range(N):
        for y_ind in range(N):
            meshpoint = mesh(x_coords[x_ind, y_ind], y_coords[x_ind, y_ind])
            try:
                gfnp[x_ind, y_ind] = abs(gfu(meshpoint))
            except Exception as e:# there are still cases where one point is not weeded out above.
                print("point failed")
                meshpoint = mesh(x_coords[x_ind-1, y_ind-1], y_coords[x_ind-1, y_ind-1])
                gfnp[x_ind, y_ind] = abs(gfu(meshpoint))

            gfnp[x_ind, y_ind] = abs(gfu(meshpoint))
    return gfnp.T #transpose bc numpy uses row major

# specific function for the plot of eigenfrequencies with respect to the mass of the electrodes to check for feasability
#TODO: do i still use this?
def plot_eigenfreqs(p, ax, eigenvals, multigfuwave, mesh, feswave, modes, color):
        mode_ind = [] # orders modes. if we have eigenvals = [20 Hz, 10 Hz, ...] with [(1,2), (1,1), ...] this array will be [1, 0, ...]
        for i in range(p["num_modes"]):
                gfumode = GridFunction(feswave)
                gfumode.vec.data = multigfuwave.vecs[i].data
                current_mode = mode_detection(p, gfumode, mesh)
                if current_mode in modes:
                        mode_ind.append(modes.index(current_mode))
        if len(mode_ind) < len(modes): # sometimes it labels the eigenfrequencies wrong, mostly not important ones so i just set them to the 1,1 freq
                for dum in range(len(modes) - len(mode_ind)):
                        print("too little mode inds")
                        mode_ind.append(0)
        # plot eigenvalues
        ax.errorbar(range(6),
                [np.sqrt(eigenvals[mode_ind[i]]) / (2 * np.pi) * 1e-3 for i in range(6)],
                yerr=[np.sqrt(eigenvals[mode_ind[i]]) / (2 * np.pi) * 1e-3 * 1e-3 for i in range(6)], color=color, linestyle = "dashed", label = "rho = " + str(p["rhoau"]))
        


# plot to comparison to the first 3 experimental results that i got
# TODO: make this general for any experimental data
def plot_comparison_experiment(p, freq_dict, mode_to_fit_to, filename, ela = True):
    mesh = generate_mesh(p)
    colors = ["blue", "green", "red"]
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xticks([1,2,3])
    modes = ["(1,1)", "(2,1)", "(3,1)"]
    ax.set_xticklabels(modes)

    eigenvecs_list, eigenvals_list = [], []
    for experiment in freq_dict.keys():
            mode_ind = []
            p["sigsin"] = freq_dict[experiment][0] * 1e6
            freq_to_fit_to = freq_dict[experiment][1][modes.index(mode_to_fit_to)]
            feswave, eigenvals, multigfuwave = solve(p, freq_to_fit_to=freq_to_fit_to, mode_to_fit_to=mode_to_fit_to, ela = ela)
            eigenvecs_list.append(multigfuwave)
            eigenvals_list.append(eigenvals)
            #reorder modes to plot the right ones
            for i in range(p["num_modes"]):
                    gfumode = GridFunction(feswave)
                    gfumode.vec.data = multigfuwave.vecs[i].data
                    current_mode = mode_detection(p, gfumode, mesh)
                    if current_mode in modes:
                            mode_ind.append(i)
            if len(mode_ind) < len(modes):
                 continue

            ax.errorbar([1,2,3],
                    [freq_dict[experiment][1][i] * 1e-3 for i in range(3)],
                    yerr=[freq_dict[experiment][2][i] * 1e-3 for i in range(3)], color=colors[experiment], label = "experiment " + str(experiment))
            ax.errorbar([1,2,3],
                    [np.sqrt(eigenvals[mode_ind[i]]) / (2 * np.pi) * 1e-3 for i in range(3)],
                    yerr=[np.sqrt(eigenvals[mode_ind[i]]) / (2 * np.pi) * 1e-3 *1e-2 for i in range(3)], color=colors[experiment], linestyle = "dashed", label = "FEM with sig = " + f'{p["sigsin"] * 1e-6:.4g}' + " MPa")
    ax.legend()
    ax.grid()
    ax.set_title("Comparison of experimental and fem results")
    ax.set_ylabel("Eigenfrequency [kHz]")
    ax.set_xlabel("resonance modes")
    plt.savefig("plots/experiment_vs_fem" + filename + ".png", bbox_inches='tight')

def plot_comparison_experiment_new(p, freq_dict, mode_to_fit_to, filename, ela = True, experiments = ["old_1", "old_2", "old_3"], modes = ["(1,1)", "(1,2)", "(2,1)","(1,3)", "(3,1)"]):
    mesh = generate_mesh(p)
    modes_all = ["(1,1)", "(1,2)", "(2,1)","(1,3)", "(3,1)"]
    mode_ind_exp = {}
    for mode in modes:
         mode_ind_exp[mode] = modes_all.index(mode)
    colors = ["blue", "green", "red", "yellow", "cyan", "purple"]
    color_dict = {}
    for i in range(len(experiments)):
         color_dict[experiments[i]] = colors[i]
    fig = plt.figure()
    ax = plt.axes()
    x = [i for i in range(len(modes))]
    ax.set_xticks(x)
    ax.set_xticklabels(modes)

    for experiment in experiments:
            p["sigsin"] = freq_dict[experiment][0] * 1e6
            if mode_to_fit_to:
                freq_to_fit_to = freq_dict[experiment][1][modes.index(mode_to_fit_to)]
            else:
                freq_to_fit_to = None
            feswave, eigenvals, multigfuwave = solve(p, freq_to_fit_to=freq_to_fit_to, mode_to_fit_to=mode_to_fit_to, ela = ela)
            write_res_to_file(p, feswave, mesh, eigenvals, multigfuwave, "None", "None", mode_to_fit_to, freq_to_fit_to)
            results = result_dict(p, mesh, feswave, multigfuwave, eigenvals)
            yfem = []
            for mode in modes:
                if mode in results.keys():
                    yfem.append(results[mode][0])
                else:
                    yfem.append(np.nan)
            y = interpolate_nans([freq_dict[experiment][1][mode_ind_exp[modes[i]]] * 1e-3 for i in range(len(modes))])
            yerr = interpolate_nans([freq_dict[experiment][1][mode_ind_exp[modes[i]]] * 1e-5 for i in range(len(modes))])
            print(yfem)
            yfem = interpolate_nans(yfem)
            yerrfem = interpolate_nans([yfem[i] *1e-2 for i in range(len(modes))])
            ax.errorbar(x,y,yerr,
                    color=color_dict[experiment], label = "experiment " + str(experiment))
            ax.errorbar(x,yfem,yerrfem,
                    color=color_dict[experiment], linestyle = "dashed", label = "FEM with sig = " + f'{p["sigsin"] * 1e-6:.4g}' + " MPa")
        
    ax.legend()
    ax.grid()
    ax.set_title("Comparison of experimental and fem results")
    ax.set_ylabel("Eigenfrequency [kHz]")
    ax.set_xlabel("resonance modes")
    plt.savefig("plots/experiment_vs_fem" + filename + ".png", bbox_inches='tight')





def convergence_analysis(p, hmaxs):
    modes = ["(1,1)", "(1,2)", "(2,1)","(1,3)", "(3,1)"]
    meshes, resultss = [], []
    colors = ["blue", "green", "red", "yellow", "cyan", "purple"]
    color_dict = {}
    for i in range(len(modes)):
         color_dict[modes[i]] = colors[i]

    for hmax in hmaxs:
        p["hmax"] = hmax
        mesh = generate_mesh(p)
        meshes.append(mesh)
        feswave, eigenvals, multigfuwave = solve(p, freq_to_fit_to=None, mode_to_fit_to=None, ela = True)
        results = result_dict(p, mesh, feswave, multigfuwave, eigenvals)
        resultss.append(results)
    return meshes, resultss

        






def write_res_to_file(p, feswave, mesh, eigenvals, multigfuwave, start_time, end_time, mode_to_fit_to, freq_to_fit_to):
    # header is:  [parameters];start_time;end_time;[eigenvals];[gfus];[mode_nrs]
    with open('result_history.csv', 'a') as file:
        para_string = ";".join(str(value) for value in p.values())
        gfus_str = "["
        mode_nrs = "["
        for i in range(len(eigenvals)):
            gf = GridFunction(feswave)
            gf.vec.data = multigfuwave.vecs[i].data
            mode_nrs += '"' + mode_detection(p, gf, mesh) + '"'
            gfu_str = str(np_gridfct(p, gf, mesh))
            gfu_str = gfu_str.replace("]\n ", "],").replace("\n ", "").replace(" ", ",")
            gfus_str += gfu_str
            if i < len(eigenvals) - 1:
                  gfus_str += ","
                  mode_nrs += ","
        gfus_str += "]"
        mode_nrs += "]"
        mode_str = mode_to_fit_to if mode_to_fit_to is not None else "None"
        freq_str = str(freq_to_fit_to) if freq_to_fit_to is not None else "None"
        eigenfreqs = [float(np.sqrt(ev) / (2 * np.pi)) for ev in eigenvals]
        file.write(";".join([para_string,start_time,end_time,mode_str,freq_str,str(eigenfreqs),gfus_str,mode_nrs]) + "\n")
def write_para_to_file(s):
    with open('para_search.txt', 'a') as file:
        file.write(s + "\n")


def draw_on_mesh(mesh, fctn = None, vectors = False):
    if fctn == None:
        scene = Draw(mesh)
        clear_output()
        scene.Draw(height="3vh")
    elif vectors:
        scene = Draw(fctn, mesh, vectors = vectors)
        clear_output()
        scene.Draw(height="3vh")
    else: #TODO: is this an ngsolve bug? when using vectors = False it still plots vectors
        scene = Draw(fctn, mesh)
        clear_output()
        scene.Draw(height="3vh")


#EXPERIMENT RUNS

#neighbourhood search algorithm to find best parameter set
def parameter_neighbourhood_search(p, freq_dict):
    mode_to_fit_to = "(2,1)" # it should not matter which mode to fit to in the end if all the fem data fits all experimental data
    mesh = generate_mesh(p)
    modes = ["(1,1)", "(2,1)", "(3,1)"]

    eigenvecs_list, eigenvals_list = [], []
    losses = {}
    paras = ['sigcr', 'rhoau', 'mpercentage_perf']
    perctgs = [0.99, 1, 1.01]
    neighbourhood = list(product(perctgs, repeat=len(paras))) # all combinations of 
    while True:
        for neighbour in neighbourhood: # search neighbourhood of best parameters so far for smaller loss.
            setup = ""
            for i in range(len(paras)):
                 p[paras[i]] *= neighbour[i]
                 setup += paras[i] + ": " + str(p[paras[i]]) + " "
            print(setup)
            key = tuple(sorted(p.items()))
            losses[key] = 0
            for experiment in freq_dict.keys():
                    print(experiment)
                    mode_ind = []
                    p["sigsin"] = freq_dict[experiment][0] * 1e6
                    freq_to_fit_to = freq_dict[experiment][1][modes.index(mode_to_fit_to)]
                    feswave, eigenvals, multigfuwave = solve(p, freq_to_fit_to = freq_to_fit_to, mode_to_fit_to = mode_to_fit_to)
                    eigenvecs_list.append(multigfuwave)
                    eigenvals_list.append(eigenvals)
                    try:
                        for i in range(p["num_modes"]):
                            gfumode = GridFunction(feswave)
                            gfumode.vec.data = multigfuwave.vecs[i].data
                            current_mode = mode_detection(p, gfumode, mesh)
                            if current_mode in modes:
                                    mode_ind.append(i)
                        if len(mode_ind) < len(modes):
                            raise Exception("did not find all the modes for experiment " + str(experiment) + " " + str(p) + " did not plot")
                        else:
                            for standard_mode_ind in range(3):
                                losses[key] += abs(freq_dict[experiment][1][standard_mode_ind] - np.sqrt(eigenvals[mode_ind[standard_mode_ind]])/(2*np.pi))
                    except Exception as e:
                        print(e)
                        losses[key] = 1e10
            for i in range(len(paras)):
                 p[paras[i]] /= neighbour[i]
        best_p_so_far = min(losses, key=losses.get) # TODO: make this more efficient. change multiple parameters at once
        print(best_p_so_far)
        print("is the best setup so far with loss = " + str(losses[best_p_so_far]))
        p = dict(best_p_so_far)




#directional search algorithm to find best parameter set
def parameter_directional_search(p, freq_dict):
    mode_to_fit_to = "(2,1)" # it should not matter which mode to fit to in the end if all the fem data fits all experimental data
    mesh = generate_mesh(p)
    modes = ["(1,1)", "(2,1)", "(3,1)"]

    eigenvecs_list, eigenvals_list = [], []
    paras = ['el_width', 'rhoau', 'sigcr']
    perct = 0.1 # steps to go in each direction
    losses = {}
    while True:
        perct /= 2 # half step size in every iteration
        for para in paras:
            for i in range(-5,5):
                setup = ""
                p[para] *= (1 + i * perct)
                setup += para + ": " + str(p[para]) + " "
                print(setup)
                write_para_to_file(setup)
                p = set_lame(p)
                key = tuple(sorted(p.items()))
                losses[key] = 0
                for experiment in freq_dict.keys():
                    mode_ind = []
                    p["sigsin"] = freq_dict[experiment][0] * 1e6
                    freq_to_fit_to = freq_dict[experiment][1][modes.index(mode_to_fit_to)]
                    feswave, eigenvals, multigfuwave = solve(p, freq_to_fit_to=freq_to_fit_to, mode_to_fit_to = mode_to_fit_to)
                    eigenvecs_list.append(multigfuwave)
                    eigenvals_list.append(eigenvals)
                    try:
                        for j in range(p["num_modes"]):
                            gfumode = GridFunction(feswave)
                            gfumode.vec.data = multigfuwave.vecs[j].data
                            current_mode = mode_detection(p, gfumode, mesh)
                            if current_mode in modes:
                                    mode_ind.append(j)
                        if len(mode_ind) < len(modes):
                            raise Exception("did not find all the modes for experiment " + str(experiment) + " " + str(p) + " did not plot")
                        else:
                            for standard_mode_ind in range(3):
                                losses[key] += abs(freq_dict[experiment][1][standard_mode_ind] - np.sqrt(eigenvals[mode_ind[standard_mode_ind]])/(2*np.pi))
                    except Exception as e:
                        print(e)
                        write_para_to_file("error")
                        losses[key] = 1e10
                p[para] /= (1 + i * perct)
            best_p_so_far = min(losses, key=losses.get) # TODO: make this more efficient. change multiple parameters at once
            print(best_p_so_far)
            write_para_to_file(str(best_p_so_far))
            print("is the best setup so far with loss = " + str(losses[best_p_so_far]))
            write_para_to_file("is the best setup so far with loss = " + str(losses[best_p_so_far]))
            p = dict(best_p_so_far)