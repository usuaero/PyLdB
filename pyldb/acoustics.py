import numpy as np

"""A perceived loudness (PLdB) calculator.

PyLdB implements Stevens' Mark VII procedure for the calculation of the
perceived loudness of a pressure signature.

Routine Listings
-----------------
PerceivedLoudness(time, pressure[, pad_f, pad_r, len_window])
Main routine. Calculates and displays perceived loudness in PLdB.



References
-----------
Fidell, S., et al. , “A first-principles model for estimating the prevalence of 
annoyance with aircraft noise exposure”, The Journal of the Acoustical Society 
of America, Vol. 130, 791, 2011.

Fidell, S., et al. , Community Response to High-Energy Impulsive Sounds: An
Assessment of the Field Since 1981”, National Research Council, 1996.
"""

def fidell_CTL(noise, growth=0.3, CTL=None, A_star=None):
    """Calculates high annoyance rate for different Community
       tolerance levels (CTL) and growth rates (0.3 for subsonic, 
       but higher for supersonic)"""
    # for A star
    if CTL is None:
        CTL = -10*np.log10(-np.log(0.5))/growth + A_star/growth
        m = (10.**(CTL/10.))**growth
        A = - m*np.log(0.5)
    else:
        A_star = growth*CTL + 10*np.log10(-np.log(0.5))
        A = 10**(A_star/10)
    m = (10.**(noise/10.))**growth
    P = np.exp(-A/m)
    return P


def DNL(Lh_day, Lh_night):
    """Calculates day-night average level (DNL) for hour equivalent
       loudness. len(Lh_day) == 15, len(Lh_night) == 9."""
    try:
        d = len(Lh_day)
    except TypeError:
        d = 15
        Lh_day = d*[Lh_day]
    try:
        n = len(Lh_night)
    except TypeError:
        n = 9
        Lh_night = n*[Lh_night]
    s = 0
    if d + n != 24:
        raise('Need 24 hours')
    for Ln in Lh_night:
        s += 10*10**(Ln/10.)
    for Ld in Lh_day:
        s += 10**(Ld/10.)
    return(10*np.log10((1/24.)*s))


def equivalent(loudnesses, sampling=None):
    """Equivalent loudness for temporal sampling. If all samples are for an
    hour, it is equivalent to hourly noise level"""
    try:
        n = len(loudnesses)
    except TypeError:
        loudnesses = sampling*[loudnesses]
        n = sampling
    s = 0
    for loudness in loudnesses:
        s += 10**(loudness/10)
    print(n)
    return 10*np.log10((1/n)*s)
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    Lh_night = 3*[55] + 3*[68] + 3*[75]
    Lh_day = 3*[86] + 3*[84] + 3*[81] + 3*[74] + 3*[69]
    PLDNL = DNL(Lh_day, Lh_night)
    print('DNL', PLDNL) # Matches from paper
    
    
    noise = np.linspace(0, 140)
    CTL = np.linspace(50, 80, 7)
    growth = np.linspace(0.3, 0.5, 3)
    plt.figure()
    for i in range(len(CTL)):
        annoyance = fidell_CTL(noise, CTL=CTL[i])
        plt.plot(noise, annoyance, label='CTL= ' + str(CTL[i]))
    plt.legend()
    plt.xlabel('PL')
    plt.ylabel('Percentage Highly Annoyed')
    plt.show()

    plt.figure()
    for i in range(len(growth)):
        annoyance = fidell_CTL(noise, CTL=75, growth=growth[i])
        plt.plot(noise, annoyance, label='Growth= %.2f' % growth[i])
    plt.legend()
    plt.xlabel('PL')
    plt.ylabel('Percentage Highly Annoyed')
    plt.show()