

ucnp_diameter = 25e-7 # cm
ucnp_density = 4.2 # g/cm^3
NAvog = 6.022e23 # mol^-1


def unitcellmolarity(x, y):
    '''NaY(1-x-y)Yb(x)Tm(y)F4'''
    M_uc = 187.9 + 84.13*x + 80.02*y # g/mol
    return(M_uc)

def REConc(x, y):
    NYb = ucnp_density * NAvog * x / unitcellmolarity(x, y)
    NTm = ucnp_density * NAvog * y / unitcellmolarity(x, y)
    print(f"{NYb=}, {NTm=}")
    return(NYb, NTm)
