import numpy as np
from celt_lpc import celt_inner_prod,xcorr_kernel,_celt_autocorr,_celt_lpc,celt_pitch_xcorr,dual_inner_prod
second_check=np.array([0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2],dtype=np.int16)

def find_best_pitch(xcorr,y,len_,max_pitch,best_pitch):
    syy=1
    best_num=np.zeros(2,dtype=np.float32)
    best_den=np.zeros(2,dtype=np.float32)
    best_num[0]=-1
    best_num[1]=-1
    best_pitch[0]=0
    best_pitch[1]=1
    for j in range(len_):
        syy+=y[j]*y[j]
    for i in range(max_pitch):
        if(xcorr[i]>0):
            xcorr16=xcorr[i]
            num =xcorr16*xcorr16
            if(num*best_den[1]>best_num[1]*syy):
                if(num*best_den[0]>best_num[0]*syy):
                    best_num[1]=best_num[0]
                    best_den[1]=best_den[0]
                    best_pitch[1]=best_pitch[0]
                    best_num[0]=num
                    best_den[0]=syy
                    best_pitch[0]=i
                else:
                    best_num[1]=num
                    best_den[1]=syy
                    best_pitch[1]=i
        syy +=y[i+len_]*y[i+len_]-y[i]*y[i]
        syy=max(syy,1)
    
def celt_fir5(x,num_,y,N,mem):
    num0=num_[0]
    num1=num_[1]
    num2=num_[2]
    num3=num_[3]
    num4=num_[4]
    mem0=mem[0]
    mem1=mem[1]
    mem2=mem[2]
    mem3=mem[3]
    mem4=mem[4]
    for i in range(N):
        s=x[i]
        s +=num0*mem0
        s +=num1*mem1
        s +=num2*mem2
        s +=num3*mem3
        s +=num4*mem4
        mem4=mem3
        mem3=mem2
        mem2=mem1
        mem1=mem0
        mem0=x[i]
        y[i]=s
    mem[0]=mem0
    mem[1]=mem1
    mem[2]=mem2
    mem[3]=mem3
    mem[4]=mem4

def pitch_downsample(x,x_lp,len_):
    ac=np.zeros(5,dtype=np.float32)
    tmp=1
    lpc=np.zeros(4,dtype=np.float32)
    mem=np.zeros(5,dtype=np.float32)
    lpc2=np.zeros(5,dtype=np.float32)
    c1=0.8
    for i in range(1,int(len_/2)):
        x_lp[i]=0.5*x[2*i]+0.25*x[2*i-1]+0.25*x[2*i+1]
    x_lp[0]=0.5*x[0]+0.25*x[1]
    _celt_autocorr(x_lp,ac,np.ones_like(x_lp),0,4,int(len_/2))
    #print("ac[1] ====",ac[1],"x_lp[10] ====",x_lp[10])
    for i in range(1,4):
        ac[i] -=ac[i]*0.008*i*0.008*i
    error=_celt_lpc(lpc,ac,4)
    
    for i in range(4):
        tmp =0.9*tmp
        lpc[i] =tmp*lpc[i]
    lpc2[0]=lpc[0]+0.8
    lpc2[1]=lpc[1]+c1*lpc[0]
    lpc2[2]=lpc[2]+c1*lpc[1]
    lpc2[3]=lpc[3]+c1*lpc[2]
    lpc2[4]=c1*lpc[3]
  #  print("lpc[10] ====",x_lp[10],"lpc2[10] ====",lpc2[1])
    celt_fir5(x_lp,lpc2,x_lp,int(len_/2),mem)


def pitch_search(x_lp,y,len_,max_pitch):
    best_pitch=np.zeros(2,dtype=np.int16)
    lag=len_+max_pitch

    x_lp4=np.zeros(int(len_/4),dtype=np.float32)
    y_lp4=np.zeros(int(lag/4),dtype=np.float32)
    xcorr=np.zeros(int(max_pitch/2),dtype=np.float32)
    for j in range(int(len_/4)):
        x_lp4[j]=x_lp[2*j]
    for j in range(int(lag/4)):
        y_lp4[j]=y[2*j]
    celt_pitch_xcorr(x_lp4,y_lp4,xcorr,int(len_/4),int(max_pitch/4))
    find_best_pitch(xcorr,y_lp4,int(len_/4),int(max_pitch/4),best_pitch)
   # print("best_pitch[0] ====",best_pitch[0],"best_pitch[1] ====",best_pitch[1])
    for i in range(int(max_pitch/2)):
        s=0
        xcorr[i]=0
        if(np.abs(i-2*best_pitch[0])>2 and np.abs(i-2*best_pitch[1])>2):
            continue
        s=celt_inner_prod(x_lp,y[i:],int(len_/2))
        xcorr[i]=max(-1,s)
    find_best_pitch(xcorr,y,int(len_/2),int(max_pitch/2),best_pitch)
    if(best_pitch[0]>0 and best_pitch[0]<int(max_pitch/2)-1):
        a=xcorr[best_pitch[0]-1]
        b=xcorr[best_pitch[0]]
        c=xcorr[best_pitch[0]+1]
        if((c-a)>(0.7*(b-a))):
            offset =1
        elif((a-c)<(0.7*(b-c))):
            offset =-1
        else:
            offset =0
    else:
        offset =0
    return 2*best_pitch[0]-offset
def compute_pitch_gain(xy,xx,yy):
    return xy/np.sqrt(1+xx*yy)

def remove_doubling(x,max_period,min_period,N,T0_,prev_period,prev_gain):
    minperiod0 = min_period;
    max_period= int(max_period/2)
    min_period= int(min_period/2)
    T0_= int(T0_/2)
    prev_period= int(prev_period/2)
    N= int(N/2)

    if(T0_>=max_period):
        T0_=max_period-1
    T=T0_
    T0=T0_

    xx,xy=dual_inner_prod(x[max_period:],x[max_period:],x[max_period-T0:],N)

    yy_lookup=np.zeros(max_period+1,dtype=np.float32)
    yy_lookup[0]=xx
    yy=xx
    for i in range(1,max_period):
        yy =yy +x[max_period-i]*x[max_period-i] -x[N+max_period-i]*x[N+max_period-i]
        yy_lookup[i]=max(0,yy)
    yy=yy_lookup[T0]
    best_xy=xy
    best_yy=yy
    g0=compute_pitch_gain(xy,xx,yy)
    g=g0
    for k in range(2,15):
        cont =0
        T1=int((2*T0+k)/(2*k))
        if(T1<min_period):
            break
        if (k==2):
            if (T1+T0>max_period):
                T1b = T0
            else:
                T1b = T0+T1;
        else:
            T1b=int((2*second_check[k]*T0+k)/(2*k))
        
        xy,xy2=dual_inner_prod(x[max_period:],x[max_period-T1:],x[max_period-T1b:],N)
        xy=0.5*(xy+xy2)
        yy=0.5*(yy_lookup[T1]+yy_lookup[T1b])
        g1=compute_pitch_gain(xy,xx,yy)
        if(np.abs(T1-prev_period)<=1):
            cont=prev_gain
        elif(np.abs(T1-prev_period)<=2 and 5*k*k<T0):
            cont=prev_gain*0.5
        else:
            cont =0
        thresh=max(0.3,0.7*g0-cont)
        if(T1<3*min_period):
            thresh=max(0.4,0.85*g0-cont)
        if(g1>thresh):
            best_xy=xy
            best_yy=yy
            g=g1
            T=T1
    
    best_xy=max(0,best_xy)
    if (best_yy<=best_xy):
        pg=1.0
    else:
       pg =best_xy/(best_yy+1)
    xcorr=np.zeros(3,dtype=np.float32)
    for k in range(3):
        xcorr[k]=celt_inner_prod(x[max_period:],x[max_period-(T+k-1):],N)
    if(xcorr[2]-xcorr[0]>(0.7*(xcorr[1]-xcorr[0]))):
        offset =1
    elif(xcorr[0]-xcorr[2]<(0.7*(xcorr[1]-xcorr[2]))):
        offset =-1
    else:
        offset =0
    if(pg>g):
        pg=g
    T0_=2*T+offset
    if (T0_<minperiod0):
      T0_=minperiod0
    return pg,T0_
  
        


