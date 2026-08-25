import numpy as np

def hz_to_mel_slaney(f):
    f=np.atleast_1d(np.asarray(f,dtype=float)).copy(); f_min=0.0; f_sp=200.0/3
    mels=(f-f_min)/f_sp
    min_log_hz=1000.0; min_log_mel=(min_log_hz-f_min)/f_sp; logstep=np.log(6.4)/27.0
    m=f>=min_log_hz
    mels[m]=min_log_mel+np.log(f[m]/min_log_hz)/logstep
    return mels if mels.size>1 else mels[0]

def mel_to_hz_slaney(m):
    m=np.atleast_1d(np.asarray(m,dtype=float)).copy(); f_min=0.0; f_sp=200.0/3
    freqs=f_min+f_sp*m
    min_log_hz=1000.0; min_log_mel=(min_log_hz-f_min)/f_sp; logstep=np.log(6.4)/27.0
    k=m>=min_log_mel
    freqs[k]=min_log_hz*np.exp(logstep*(m[k]-min_log_mel))
    return freqs

def mel_filterbank(sr,n_fft,n_mels,fmin=0.0,fmax=None):
    if fmax is None: fmax=sr/2
    fftfreqs=np.fft.rfftfreq(n_fft,1.0/sr)
    mel_pts=np.linspace(hz_to_mel_slaney(fmin),hz_to_mel_slaney(fmax),n_mels+2)
    freqs=mel_to_hz_slaney(mel_pts)
    fdiff=np.diff(freqs)
    ramps=freqs.reshape(-1,1)-fftfreqs.reshape(1,-1)
    W=np.zeros((n_mels,len(fftfreqs)))
    for i in range(n_mels):
        lower=-ramps[i]/fdiff[i]; upper=ramps[i+2]/fdiff[i+1]
        W[i]=np.maximum(0,np.minimum(lower,upper))
    enorm=2.0/(freqs[2:n_mels+2]-freqs[:n_mels])   # slaney norm
    W*=enorm[:,None]
    return W

def stft_power(y,n_fft,hop,center=True,pad_mode='constant'):
    win=np.hanning(n_fft+1)[:-1]                    # scipy/librosa 'hann' periodic
    if center: y=np.pad(y,n_fft//2,mode='constant')
    n=1+(len(y)-n_fft)//hop
    idx=np.arange(n_fft)[None,:]+hop*np.arange(n)[:,None]
    frames=y[idx]*win[None,:]
    S=np.fft.rfft(frames,n=n_fft,axis=1)
    return (np.abs(S)**2).T                          # (bins, frames)

def power_to_db(S,ref=None,amin=1e-10,top_db=80.0):
    S=np.asarray(S); ref=np.max(S) if ref is None else ref
    d=10.0*np.log10(np.maximum(amin,S))-10.0*np.log10(np.maximum(amin,ref))
    if top_db is not None: d=np.maximum(d,d.max()-top_db)
    return d

def logmel(y,sr,n_fft,hop,n_mels,fmin=0.0,fmax=None):
    P=stft_power(y,n_fft,hop)
    W=mel_filterbank(sr,n_fft,n_mels,fmin,fmax)
    M=W@P
    return power_to_db(M,ref=None,top_db=80.0)
