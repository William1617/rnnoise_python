import numpy as np
import onnxruntime
import h5py
import soundfile as sf
from pitch import pitch_search,pitch_downsample,remove_doubling
g_frame_size=160
g_hop_size=80
def biquad(y,mem,x,b,a,N):
    for i in range(N):
        xi=x[i]
        yi=x[i]+mem[0];
        mem[0]=mem[1] + (b[0]*xi - a[0]*yi)
        mem[1]=b[1]*xi - a[1]*yi
        y[i]=yi


class Rnnoise16k():
    def __init__(self,training,model_path=None):
        self.training=training
        if(self.training==False and model_path is not None):
            self.model=onnxruntime.InferenceSession(model_path)
            self.model_input_name=[inp.name for inp in self.model.get_inputs()]
            self.model_input = {}
            self.model_input[self.model_input_name[0]]=np.zeros((1,1,38),dtype=np.float32)
            self.model_input[self.model_input_name[1]]=np.zeros((1,1,24),dtype=np.float32)
            self.model_input[self.model_input_name[2]]=np.zeros((1,1,48),dtype=np.float32)
            self.model_input[self.model_input_name[3]]=np.zeros((1,1,96),dtype=np.float32)
            
        self.frame_size=g_frame_size
        self.freq_size=self.frame_size +1
        self.window_size=2*self.frame_size
        self.hop_size=g_hop_size
        self.nb_bands=18
        self.ceps_mem=8
        self.nb_delta_ceps=6
        self.nb_features=self.nb_bands+self.nb_delta_ceps*3+2
        self.pitch_frame_size=320
        self.pitch_max_period=256
        self.pitch_buf_size=self.pitch_frame_size+self.pitch_max_period
        self.pitch_min_period=20
       
        self.analysis_mem=np.zeros(self.frame_size).astype(np.float32)
        self.cepstral_mem=np.zeros((self.ceps_mem,self.nb_bands)).astype(np.float32)
        self.memid=0
        self.synthesis_mem=np.zeros(self.frame_size).astype(np.float32)
        self.pitch_buf=np.zeros(self.pitch_buf_size).astype(np.float32)
        self.pitch_enh_buf=np.zeros(self.pitch_buf_size).astype(np.float32)
        self.last_gain=0
        self.last_period=0
        self.mem_hp_x=np.zeros(2).astype(np.float32)
        self.lastg=np.zeros(self.nb_bands).astype(np.float32)
        self.window=[]
        self.dct_table=np.zeros(self.nb_bands*self.nb_bands).astype(np.float32)
        self.count=0  
        
        
        self.eband5ms=np.array([0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100],dtype=np.int16)
        pi=3.14159265358979323846
        for i in range(self.frame_size):
            self.window.append(np.sin(0.5*pi*np.sin(0.5*pi*(i+0.5)/self.frame_size)*np.sin(0.5*pi*(i+0.5)/self.frame_size)))
        for i in range(self.nb_bands):
            for j in range(self.nb_bands):
                self.dct_table[i*self.nb_bands+j]=np.cos((i+0.5)*j*np.pi/self.nb_bands)
                if(j==0):
                    self.dct_table[i*self.nb_bands+j]=np.sqrt(0.5)
    def reset_mem(self):
        self.analysis_mem=np.zeros(self.frame_size).astype(np.float32)
        self.cepstral_mem=np.zeros((self.ceps_mem,self.nb_bands)).astype(np.float32)
        self.memid=0
        self.synthesis_mem=np.zeros(self.frame_size).astype(np.float32)
        self.pitch_buf=np.zeros(self.pitch_buf_size).astype(np.float32)
        self.pitch_enh_buf=np.zeros(self.pitch_buf_size).astype(np.float32)
        self.last_gain=0
        self.last_period=0
        self.mem_hp_x=np.zeros(2).astype(np.float32)
        self.lastg=np.zeros(self.nb_bands).astype(np.float32)
       
        
    def compute_band_energy(self,bandE,X):
        tmp_sum=np.zeros(self.nb_bands).astype(np.float32)
        for i in range(self.nb_bands-1):
            band_size=(self.eband5ms[i+1]-self.eband5ms[i])*4
            for j in range(band_size):
                frac=j/band_size
                curren_idx=self.eband5ms[i]*4+j
                tmp=X[curren_idx].real*X[curren_idx].real+X[curren_idx].imag*X[curren_idx].imag
                tmp_sum[i]+=(1-frac)*tmp
                tmp_sum[i+1]+=frac*tmp
        tmp_sum[0]*=2
        tmp_sum[self.nb_bands-1]*=2
        for i in range(self.nb_bands):
            bandE[i]=tmp_sum[i]
    def compute_band_corr(self,bandCorr,X,Y):
        tmp_sum=np.zeros(self.nb_bands).astype(np.float32)
        for i in range(self.nb_bands-1):
            band_size=(self.eband5ms[i+1]-self.eband5ms[i])*4
            for j in range(band_size):
                frac=j/band_size
                curren_idx=self.eband5ms[i]*4+j
                tmp=X[curren_idx].real*Y[curren_idx].real+X[curren_idx].imag*Y[curren_idx].imag
                tmp_sum[i]+=(1-frac)*tmp
                tmp_sum[i+1]+=frac*tmp
        tmp_sum[0]*=2
        tmp_sum[self.nb_bands-1]*=2
        for i in range(self.nb_bands):
            bandCorr[i]=tmp_sum[i]
    def interp_band_gain(self,g,bandE):
        g[:]=0
        for i in range(self.nb_bands-1):
            band_size=(self.eband5ms[i+1]-self.eband5ms[i])*4
            for j in range(band_size):
                frac=j/band_size
                g[self.eband5ms[i]*4+j]=(1-frac)*bandE[i]+frac*bandE[i+1]
        return g
    def frame_analysis(self,X,Ex,in_data):
        x=np.zeros(self.window_size).astype(np.float32)
        x[:self.frame_size]=self.analysis_mem
        x[self.frame_size:]=in_data
        self.analysis_mem=in_data
        x=self.apply_window(x)
        x_fft=np.fft.rfft(x)/320
        for i in range(self.freq_size):
            X[i]=x_fft[i]
        if(self.training):
            X[self.frame_size+1:self.freq_size]=0

        self.compute_band_energy(Ex,X)
      
    def frame_synthesis(self,out_data,y):
        
        y_fft=np.fft.irfft(y)*320
        y_fft=self.apply_window(y_fft)
        
        for i in range(self.frame_size):
            out_data[i]=y_fft[i] +self.synthesis_mem[i]
        self.synthesis_mem[:]=y_fft[self.frame_size:]
    def apply_window(self,x):
        for i in range(self.frame_size):
            x[i]=x[i]*self.window[i]
            x[self.window_size-i-1]=x[self.window_size-i-1]*self.window[i]
        return x
    def dtc(self,out_,in_):
        for i in range(self.nb_bands):
            sum=0
            for j in range(self.nb_bands):
                sum+=in_[j]*self.dct_table[j*self.nb_bands+i]
            out_[i]=sum*np.sqrt(2.0/22)
    
    def pitch_filter(self,X,P,Ex,Ep,Exp,g):
        r=np.zeros(self.nb_bands).astype(np.float32)
        rf=np.zeros(self.freq_size).astype(np.float32)
        for i in range(self.nb_bands):
            if(Exp[i]>g[i]):
                r[i]=1
            else:
                r[i]=Exp[i]*Exp[i]*(1-g[i]*g[i])/(0.001+g[i]*g[i]*(1-Exp[i]*Exp[i]))
            r[i]=np.sqrt(min(1,max(0,r[i])))
            r[i]*=np.sqrt(Ex[i]/(1e-8+Ep[i]))
        rf=self.interp_band_gain(rf,r)
        for i in range(self.freq_size):
            X[i] +=rf[i]*P[i]

        newE=np.zeros(self.nb_bands).astype(np.float32)
        self.compute_band_energy(newE,X)
        norm=np.zeros(self.nb_bands).astype(np.float32)
        normf=np.zeros(self.freq_size).astype(np.float32)
        for i in range(self.nb_bands):
            norm[i]=np.sqrt(Ex[i]/(1e-8+newE[i]))
        self.interp_band_gain(normf,norm)
        X=X*normf
   
    def compute_frame_features(self,X,P,Ex,Ep,Exp,features,input_data):
        E=0
        ceps_0=np.zeros(self.nb_bands).astype(np.float32)
        ceps_1=np.zeros(self.nb_bands).astype(np.float32)
        ceps_2=np.zeros(self.nb_bands).astype(np.float32)
        spec_variability=0
        Ly=np.zeros(self.nb_bands).astype(np.float32)
        p=np.zeros(self.window_size).astype(np.float32)
        pitch_buf=np.zeros(int(self.pitch_buf_size/2)).astype(np.float32)
        tmp=np.zeros(self.nb_bands).astype(np.float32)
        self.frame_analysis(X,Ex,input_data)
        self.pitch_buf[:self.pitch_buf_size-self.frame_size]=self.pitch_buf[self.frame_size:]
        self.pitch_buf[self.pitch_buf_size-self.frame_size:]=input_data
        pre=self.pitch_buf[0:]
       
        pitch_downsample(pre,pitch_buf,self.pitch_buf_size)
        pitch_index=pitch_search(pitch_buf[int(self.pitch_max_period/2):],pitch_buf,self.pitch_frame_size,self.pitch_max_period-3*self.pitch_min_period)
        
        pitch_index=self.pitch_max_period-pitch_index
    
        gain,pitch_index=remove_doubling(pitch_buf,self.pitch_max_period,self.pitch_min_period,self.pitch_frame_size,pitch_index,self.last_period,self.last_gain)
      
        self.last_gain=gain
        self.last_period=pitch_index
        for i in range(self.window_size):
            p[i]=self.pitch_buf[self.pitch_buf_size-self.window_size-pitch_index+i]
        p=self.apply_window(p)
        p_fft =np.fft.rfft(p)
        for i in range(self.freq_size):
            P[i]=p_fft[i]/320
        self.compute_band_energy(Ep,P)
        self.compute_band_corr(Exp,X,P)
    
        Exp /=np.sqrt(Ep*Ex+0.001)
        self.dtc(tmp,Exp)
        
        features[self.nb_bands+2*self.nb_delta_ceps:self.nb_bands+3*self.nb_delta_ceps]=tmp[:self.nb_delta_ceps]
        features[self.nb_bands+2*self.nb_delta_ceps] -=1.3
        features[self.nb_bands+2*self.nb_delta_ceps+1] -=0.9
        features[self.nb_bands+3*self.nb_delta_ceps]=0.01*(pitch_index-300)
        logMax=-2
        follow=-2
        for i in range(self.nb_bands):
            Ly[i]=np.log10(1e-2+Ex[i])
            Ly[i]=max(logMax-7,max(follow-1.5,Ly[i]))
            logMax=max(logMax,Ly[i])
            follow=max(follow-1.5,Ly[i])
            E +=Ex[i]
        if (self.training==False and E<0.04):
            features[:]=0
            return 1
        self.dtc(features,Ly)
        features[0] -=12
        features[1] -=4
        self.cepstral_mem[self.memid]=features[:self.nb_bands]
        if(self.memid<1):
            ceps_id1=self.ceps_mem-1+self.memid
        else:
            ceps_id1=self.memid-1
        if(self.memid<2):
            ceps_id2=self.ceps_mem-2+self.memid
        else:
            ceps_id2=self.memid-2
       
        
        for i in range(self.nb_delta_ceps):
            features[i]=self.cepstral_mem[self.memid][i]+self.cepstral_mem[ceps_id1][i]+self.cepstral_mem[ceps_id2][i]
            features[self.nb_bands+i]=self.cepstral_mem[self.memid][i]-self.cepstral_mem[ceps_id2][i]
            features[self.nb_bands+self.nb_delta_ceps+i]=self.cepstral_mem[self.memid][i]-2*self.cepstral_mem[ceps_id1][i]+self.cepstral_mem[ceps_id2][i]
        self.memid=(self.memid+1)%self.ceps_mem
        for i in range(self.ceps_mem):
            
            mindst=1e10
            for j in range (self.ceps_mem):
                dist=0
                for k in range(self.nb_bands):
                    tmp =self.cepstral_mem[i][k]-self.cepstral_mem[j][k]
                    dist +=tmp*tmp
                if (j!=i):
                    mindst=min(mindst,dist)
            spec_variability+=mindst
        features[self.nb_bands+3*self.nb_delta_ceps+1]=spec_variability/(self.ceps_mem-2.1)
        
        return self.training and E<0.1

    def rnnoise_process_frame(self,input_data,output_data):
        if(self.training==True):
            return 0
        x=np.zeros(self.frame_size).astype(np.float32)
        X=np.zeros(self.freq_size).astype(np.complex64)
        P=np.zeros(self.window_size).astype(np.complex64)
        ex=np.zeros(self.nb_bands).astype(np.float32)
        ep=np.zeros(self.nb_bands).astype(np.float32)
        exp=np.zeros(self.nb_bands).astype(np.float32)
        features=np.zeros(self.nb_features).astype(np.float32)
        g=np.ones(self.nb_bands).astype(np.float32)
        gf=np.ones(self.freq_size).astype(np.float32)
        vad_prob=0
        silence=0
        a_hp=np.array([-1.99599, 0.99600],dtype=np.float32)
        b_hp=np.array([-2, 1],dtype=np.float32)
        biquad(x,self.mem_hp_x,input_data,b_hp,a_hp,self.frame_size)
      
        silence=self.compute_frame_features(X,P,ex,ep,exp,features,x)
    
        self.count+=1
        if(silence==False):
            self.model_input[self.model_input_name[0]]=np.reshape(features,(1,1,38))
            self.model_output=self.model.run(None,self.model_input)
            for i in range(3):
               self.model_input[self.model_input_name[i+1]]=self.model_output[i+2]
            g=np.squeeze(self.model_output[0])
        
            vad_prob=np.squeeze(self.model_output[1])
            self.pitch_filter(X,P,ex,ep,exp,g)
            for i in range(self.nb_bands):
                alpha=0.6
                g[i]=max(g[i],alpha*self.lastg[i])
                self.lastg[i]=g[i]
            gf=self.interp_band_gain(gf,g)
            for i in range(self.freq_size):
                X[i]*=gf[i]
     
        self.frame_synthesis(output_data,X)
        
        return vad_prob

    
if __name__ == "__main__":
    rnnoise=Rnnoise16k(False,"rnnoise16k.onnx")
    audio_name='input.wav'
    audio,fs=sf.read(audio_name)
  
    output_audio=np.zeros(len(audio)).astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
 
    block_num=len(audio)//g_frame_size-1
    for i in range(block_num):
        input_data=audio[i*g_frame_size:i*g_frame_size+g_frame_size]*32768
        output_data=np.zeros(g_frame_size).astype(np.float32)
        vad_prob=rnnoise.rnnoise_process_frame(input_data,output_data)
        audio[i*g_frame_size:i*g_frame_size+g_frame_size]=output_data/32768
    sf.write('output.wav',audio,fs)
   
