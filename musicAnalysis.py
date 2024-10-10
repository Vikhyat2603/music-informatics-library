"""
Acknowledgement:

Parts of code were adapted from:
CMSC 395 Music Informatics Lab & HW assignments (taught by Dr. Yucong Jiang)
"""

import librosa
import numpy as np
import scipy
import matplotlib.pyplot as plt
import IPython.display as ipd

defaultSR = 8000

# lab 2
def loadAudio(filename, sr=None):
    """return a tuple of the samples array and the sample rate

    Parameters:
    filename: file name for audio
    sr: sampling rate (default is 8000, changes to SR of any audio when loaded)
    """
    
    thisWav, thisSR = librosa.load(filename, sr=sr)
    if sr is None:
        defaultSR = thisSR
    return thisWav, thisSR

def playAudio(wave, sr=defaultSR, normalize=True):
    """return an IPD Audio object of the wave

    Parameters:
    wave: array of sample values
    sr: sampling rate (default is SR of last audio loaded, or 8000 if none loaded)
    normalize: whether to normalize sample values to [-1,1]
    """
    
    return ipd.display(ipd.Audio(data=wave, rate=sr, normalize=normalize))

def plotAudio(wave, sr=defaultSR, xAxisTime=False, title='Audio Samples'):
    """plot & show the wave

    Parameters:
    wave: array of sample values
    sr: sampling rate (default is SR of last audio loaded, or 8000 if none loaded)
    xAxisTime: whether to convert x axis labels to time values (based on sr)
    title: title of plot    
    """
    
    endTime = len(wave)/sr 
    t = np.arange(0, len(wave))
    
    if xAxisTime:
        t = np.linspace(0, endTime, len(wave))
    
    plt.plot(t, wave)
    plt.xlabel('Time (s)' if xAxisTime else 'Sample')
    plt.ylabel('Sample value')
    plt.title(title)
    plt.show()

# lab 3
def synthesiseNote(freq, duration, sr=defaultSR):
    """return tuple of time and sample values for a pure note of given freq
    and duration

    Parameters:
    freq: frequency (in Hz) of note to be synthesised
    duration: duration (in seconds) of note to be synthesised
    sr: sampling rate (default is SR of last audio loaded, or 8000 if none loaded)
    """
    
    times = np.linspace(0, duration, int(sr*duration)+1) # create time step values
    samples = np.sin(2*np.pi*freq*times) # evaluate the pure note function on each time step
    return times, samples   

# lab 4
def freqToMIDI(freq):
    """return MIDI number (float) corresponding to the given frequency (in Hz)

    Parameters:
    freq: frequency (in Hz)
    """

    MIDI = 69 + 12*np.log2(freq/440) # convert frequency to MIDI (float)
    return MIDI

def MIDItoFreq(MIDI):
    """return frequency (in Hz) corresponding to the given MIDI number

    Parameters:
    MIDI: MIDI number
    """

    freq = 440*np.power(2, (MIDI-69)/12) # convert MIDI number to some freq
    return freq

# lab 5
def smoothGlissandoFreqW(freqA, freqB, duration, sr=defaultSR):
    """return array of frequencies (in Hz) for glissando of linearly paced freq

    Parameters:
    freqA: starting frequency (in Hz)
    freqB: ending frequency (in Hz)
    duration: duration of glissando (in seconds)
    sr: sampling rate (default is SR of last audio loaded, or 8000 if none loaded)
    """
    
    t = np.linspace(0, 1, int(sr*duration)+1)
    freqArr = freqA + (freqB-freqA)*t
    return freqArr

def uniformGlissandoFreqW(freqA, freqB, duration, sr=defaultSR):
    """return array of frequencies (in Hz) for glissando of exponentially paced freq

    Parameters:
    freqA: starting frequency (in Hz)
    freqB: ending frequency (in Hz)
    duration: duration of glissando (in seconds)
    sr: sampling rate (default is SR of last audio loaded, or 8000 if none loaded)
    """
    
    t = np.linspace(0, 1, int(sr*duration)+1)
    freqArr = freqA * np.power(freqB/freqA,t)
    return freqArr

def getVibratoFreqWave(baseFreq, vibratoDuration, W, v, sr=defaultSR):
    '''frequency of base note, duration of vibrato, sample rate,
    amplitude of frequency vibrato, frequency of added vibrato'''

    """return array of frequencies (in Hz) for vibrato

    Parameters:
    baseFreq: frequency of base note (in Hz)
    vibratoDuration: duration of vibrato (in seconds)
    W: amplitude of frequency oscillation in viibrato (in Hz)
    v: frequency of oscillation of vibrato (in Hz)
    sr: sampling rate (default is SR of last audio loaded, or 8000 if none loaded)
    """
    
    vibratoSamples = int(vibratoDuration * sr) # + 1 is not included to avoid length mismatches

    t = np.linspace(0, vibratoDuration, vibratoSamples)
    vibratoFreqArr = np.repeat(baseFreq, vibratoSamples) + W * np.sin(2*np.pi*v*t) # take base note and add vibrato perturbation

    return vibratoFreqArr

def freqWaveToWave(freqArr, sr=defaultSR):
    """return wave sample values by calculating numerical integral of frequency array

    Parameters:
    freqArr: array of frequencies
    sr: sampling rate (default is SR of last audio loaded, or 8000 if none loaded)
    """
    
    return np.sin(2*np.pi*np.cumsum(freqArr)/sr)

# lab 7
def fft(wave):
    """return Fast Fourier Transform of wave

    Parameters:
    wave: array of sample values
    """
    
    return np.fft.fft(wave)

def ifft(ft):
    """return Inverse Fast Fourier Transform of wave

    Parameters:
    ft: fourier transform of wave
    """
    
    return np.fft.ifft(ft)

# lab 8
def convolve(a, b):
    """return convolution result of two arrays

    Parameters:
    a: array 1 (wave)
    b: array 2 (filter)
    """

    return scipy.signal.convolve(a, b, mode='same')
  
# lab 9
def getSTFT(wave, frameSize, hopSize, sr=defaultSR, applyHannWindow=False):
    """return Short Time Fourier Transform of wave

    Parameters:
    wave: array of sample values
    frameSize: size of one frame (number of sample values in it)
    hopSize: jump size between successive frames (number of sample values between them)
    sr: sampling rate (default is SR of last audio loaded, or 8000 if none loaded)
    applyHannWindow: whether to apply Hann windowing to each frame before taking FFT
    """
    
    nFrames = int( (len(wave)-frameSize) / hopSize) + 1 # number of frames in wave

    dftLength = int(frameSize/2) + 1 # number of frequency bins in a frame
    stft = np.zeros((dftLength, nFrames), dtype='complex') # initialise array for STFT
    hannWindow = 0.5 - 0.5*np.cos(2*np.pi*np.arange(frameSize)/frameSize) # hann window matching frame size
    
    for frameIndex in range(nFrames):
        # extract the frameIndex-th frame
        waveFrame = np.copy(wave[hopSize*frameIndex : hopSize*frameIndex + frameSize])

        # apply the hann window if enabled            
        if applyHannWindow:
            waveFrame *= hannWindow

        # store half-sliced FFT of this frame as a column of STFT
        stft[:, frameIndex] = fft(waveFrame)[:dftLength]
        
    return stft

def plotSTFT(stft, frameSize=None, hopSize=None, sr=defaultSR, timeOnXAxis=False,
             freqOnYAxis=False, decibelScale=False, showPlot=True, title='Spectogram'):

    """return Short Time Fourier Transform of wave

    Parameters:
    stft: Short Time Fourier Transform
    frameSize: size of one frame (number of sample values in it)
    sr: sampling rate (default is SR of last audio loaded, or 8000 if none loaded)
    xAxisTime: whether to convert x axis labels to time values (based on sr, hopSize, & frameSize)
    yAxisTime: whether to convert y axis labels to frequency values (based on sr & frameSize)
    decibelScale: whether to convert STFT element-wise color values to log scale
    showPlot: whether to show plot at end of function
    title: title of plot
    """
    
    spect = np.abs(stft)

    yMax = stft.shape[0] # number of frequency bins
    xMax = stft.shape[1] # number of frames

    yLabel = 'Frequency Bin (k)'
    xLabel = 'Frame Index'

    # convert frame indexes on x-axis to time-values if enabled
    if timeOnXAxis:
        assert (hopSize is not None) and (frameSize is not None),\
            "hopSize and frameSize must be provided for x-axis time value calculations"
            
        xLabel = 'Time (seconds)'
        nSamples = (xMax-1)*hopSize + frameSize
        xMax = nSamples/sr
        plt.xlim([0, xMax])

    # convert frequency bin numbers on y-axis to frequencies if enabled    
    if freqOnYAxis:
        assert (frameSize is not None),\
            "frameSize must be provided for y-axis frequnecy value calculations"
            
        yLabel = 'Frequency (Hz)'
        yMax = yMax * sr/frameSize # k_max * SR/N
        plt.ylim([0, yMax])

    # convert STFT's elements to log scale if enabled
    if decibelScale:
        spectDB = np.log10(1e-9 + spect) # small value added to avoid log(0) error
        plt.imshow(spectDB, origin='lower', aspect='auto',
                   extent=[0,  xMax, 0, yMax])
    else:
        plt.imshow(spect, origin='lower', aspect='auto',
                   extent=[0,  xMax, 0, yMax])

    # set labels and titles for plot
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.colorbar(label=('Magnitude (dB)' if decibelScale else 'Magnitude'))

    # show plot if enabled
    if showPlot:
        plt.show()

# lab 10

def getTemplate(midi, frameSize, sr=defaultSR):
    """return a frequency profile template for the given midi note

    midi: midi number of note
    frameSize: size of one frame (number of sample values in it)
    sr: sampling rate (default is SR of last audio loaded, or 8000 if none loaded)
    """
    assert frameSize % 2 == 0, 'Frame size should be even!'

    f0 = MIDItoFreq(midi) # fundamental frequency
    H = 13 # maximumn number of harmonics to include
    bins = 1 + frameSize//2 # number of frequency bins
    
    n = np.arange(0, bins) # frequency bin indexes

    # function to find one harmonic peak with center k (calculate its magnitude over n bins)
    func = lambda k: np.exp(-0.01*n - 0.5*np.square((n-k)/(0.01*k+0.01)))

    # sum all harmonics of the base note to define the template for this note
    template = np.array(sum(func(round(f0*h*frameSize/sr)) for h in range(1,H+1)))
    
    template += 0.002 # to avoid zeros
    template /= template.sum() # to normalize sum to 1 (template ~= prob distrubtion)
    return template

def createTemplates(frameSize, sr=defaultSR, midiStart=24, midiEnd=120):
    """return a dictionary with midi numbers (or 0) as key and template of this
    midi note as value. (silence is represented by MIDI 0)

    Parameters:
    frameSize: size of one frame (number of sample values in it)
    sr: sampling rate (default is SR of last audio loaded, or 8000 if none loaded)
    midiStart: MIDI number for lowest note to create a template for
    midiEnd: MIDI number for highest note to create a template for
    """
    
    templates = {}
    for midi in range(midiStart, midiEnd + 1):
        templates[midi] = getTemplate(midi, frameSize, sr)
        
    templates[0] = np.repeat(1/(1+frameSize//2), 1+frameSize//2) # template for silence
    return templates

def getLogLikelihoods(templates, stft, midiStart=24, midiEnd=120):
    """calculate and return a 2D log-likelihoods matrix of observing each frame in
    the spectogram given each template. Row indices represent midi numbers, and
    column indices are the same as the frame indices.
    From Row 1 to Row (LOW_MIDI-1), all entries are zero because they are outside of
    our midi range (LOW_MIDI to HIGH_MIDI) and zero (representing silence/rest).

    Parameters:
    templates: dictionary with MIDI numbers and corresponding templates as keys & values
    stft: Short Time Fourier Transform
    midiStart: MIDI number for lowest note to calculate log-likelihood for
    midiEnd: MIDI number for highest note to calculate log-likelihood for

    Uses linear algebra approach to finding likelihood matrix without individual calculate_log_likelihood calls        
    """
    spectrogram = np.abs(stft)
    
    frames = spectrogram.shape[1] # the total number of frames
    likelihoods = np.zeros((midiEnd+1, frames)) # initialise 2D array for likelihood matrix
    
    halfN = spectrogram.shape[0] # DFT length
    silenceTemplate = templates[0] # template for silence

    # get template for each MIDI number between midiStart and midiEnd
    midiTemplates = np.array([templates[i] for i in range(midiStart, midiEnd+1)])

    # stack silenceTemplate with templates for each included MIDI note, find log
    templateMatrix = np.log(np.vstack([silenceTemplate, midiTemplates]) + 1e-8)

    # normalise spectogram such that each column sums to 1
    # (each frame slice is a prob density function)
    normalisedSpect = spectrogram / (1e-8 + spectrogram.sum(axis=0))

    # calculate log-likelihoods for each template and frame
    likelihoods = np.dot(templateMatrix, normalisedSpect)

    # normalise log-likehoods to the range [0,1] (just for standardisation,
    # even though they are log-likelihoods
    likelihoods -= likelihoods.min()
    likelihoods /= likelihoods.max()

    # create empty 2D array for missing templates for MIDI notes not included
    emptyResults = np.zeros((midiStart-1, frames))

    # add these empty templates to likelihoods matrix
    likelihoods = np.vstack([likelihoods[0], emptyResults, likelihoods[1:]])        

    return likelihoods
    
def identifyLikelyNotes(likelihoods, considerSilence=True):
    """calculate the most probable note (or silence) for each frame

    Parameters:
    likelihoods: matrix containing the log-likelihoods for each MIDI number (in the
                 range [midiStart, midiEnd]) for each frame
    considerSilence: whether to consider the possibility of silent notes
    """

    # check most likely note including silence in some frames if enabled
    if considerSilence:        
        # Find the MIDI of the note (or silence) with highest likelihood
        likelyNotes = likelihoods.argmax(axis=0)
    else:
        # Find the MIDI of the note with highest likelihood
        likelyNotes = likelihoods[1:].argmax(axis=0)+1

##        semitoneTolerance = 1 # maximum sudden jump lasting one frame
##        likelyNotes = removeSpikes(likelyNotes, semitoneTolerance)

    return likelyNotes

def removeSpikes(array, tolerance=1):
    """remove erroneous 'spikes' from array. 'Spikes' are points which are more than T
    units vertically distant from their adjacent points, T being a tolerance variable

    Parameters:
    array: array from which 'spikes' are to be removed
    tolerance: minimum height of a 'spike' for it to be removed
    """

    # for each note except the first & last notes
    for i in range(1,len(array)-1):
        
        # find average MIDIs of left and right adjacent notes of this note
        nbrAvg = (array[i-1]+array[i+1])/2 

        # if this note's MIDI is v diff from each neighbouring MIDI
        if (abs(array[i-1] - array[i]) > tolerance) and\
            (abs(array[i+1] - array[i]) > tolerance):
            
            # set this note MIDI to that average MIDI
            array[i] = int(nbrAvg)

    return array
            
def getSpectralCentroid(stft, frameSize, sr=defaultSR):
    """calculates the spectral centroid (expected value of frequency (Hz)) for each frame

    Parameters:
    stft:
    frameSize: size of one frame (number of sample values in it)
    sr: sampling rate (default is SR of last audio loaded, or 8000 if none loaded)
    """

    # list of frequencies (Hz) corresponding corresponding to each frequency bin
    freqs = np.arange(stft.shape[0])*sr/frameSize

    # the list of frequencies repeated nFrames number of times and then transposed
    # to produce a matrix of the same shape as STFT, with all columns being the same
    freqsMat = np.tile(freqs, (stft.shape[1],1)).T

    # magnitude of DFT 
    stftMag = np.abs(stft)

    # calculate weighted average of the different frequencies for each frame
    spectralCentroid = np.abs(np.sum(freqsMat * stftMag, axis=0) / (np.sum(stftMag, axis=0)+1e-8))
    return spectralCentroid
