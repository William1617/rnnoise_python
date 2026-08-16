import numpy as np

def xcorr_kernel(x,y,sum,len_):
    if(len_<3):
        print("xcorr_kernel: len < 3")
        return
    y_3=0
    y_0=y[0]
    y_1=y[1]
    y_2=y[2]
    j=3
    while(j<len_):
        tmp=x[j-3]
        y_3=y[j]
        sum[0] +=tmp*y_0
        sum[1] +=tmp*y_1
        sum[2] +=tmp*y_2
        sum[3] +=tmp*y_3
        j +=1
        tmp=x[j-3]
        y_0=y[j]
        sum[0] +=tmp*y_1
        sum[1] +=tmp*y_2
        sum[2] +=tmp*y_3
        sum[3] +=tmp*y_0
        j +=1
        tmp=x[j-3]
        y_1=y[j]
        sum[0] +=tmp*y_2
        sum[1] +=tmp*y_3
        sum[2] +=tmp*y_0
        sum[3] +=tmp*y_1
        j +=1
        tmp=x[j-3]
        y_2 =y[j]
        sum[0] +=tmp*y_3
        sum[1] +=tmp*y_0
        sum[2] +=tmp*y_1
        sum[3] +=tmp*y_2
        j +=1
    if (j<len_-1):
        tmp=x[j-3]
        y_3=y[j]
        sum[0] +=tmp*y_0
        sum[1] +=tmp*y_1
        sum[2] +=tmp*y_2
        sum[3] +=tmp*y_3
        j +=1
    if (j<len_-1):
        tmp=x[j-3]
        y_0=y[j]
        sum[0] +=tmp*y_1
        sum[1] +=tmp*y_2
        sum[2] +=tmp*y_3
        sum[3] +=tmp*y_0
        j +=1
    if (j<len_):
        tmp=x[j-3]
        y_1=y[j]
        sum[0] +=tmp*y_2
        sum[1] +=tmp*y_3
        sum[2] +=tmp*y_0
        sum[3] +=tmp*y_1
        j +=1

def dual_inner_prod(x,y01,y02,N):
    xy01=0
    xy02=0
    for i in range(N):
        xy01 +=x[i]*y01[i]
        xy02 +=x[i]*y02[i]
    return xy01,xy02
def celt_inner_prod(x,y,N):
    xy=0
    for i in range(N):
        xy +=x[i]*y[i]
    return xy

def celt_pitch_xcorr(_x,_y,xcorr,len_,max_pitch):
    i = 0
    while i < max_pitch - 3:
        sum4 = np.zeros(4, dtype=np.float32)
        xcorr_kernel(_x,_y[i:],sum4,len_)
        xcorr[i] = sum4[0]
        xcorr[i+1] = sum4[1]
        xcorr[i+2] = sum4[2]
        xcorr[i+3] = sum4[3]
        i += 4
    while i < max_pitch:
        s = 0.0
        s =celt_inner_prod(_x, _y[i:], len_)
        xcorr[i] = s
        i += 1
    return xcorr

def _celt_lpc(lpc,ac,p):
   
    if ac[0] == 0:
        lpc[:p] = 0
        return 0.0

    a = np.zeros(p, dtype=np.float64)
    e = ac[0]

    for i in range(p):
        # sum_k = sum(a[j] * ac[i - j] for j in range(i))
        # Note: ac[i:0:-1] is empty when i==1, and a[i-1::-1] is wrong when i==0
        sum_k = 0.0
        for j in range(i):
            sum_k += a[j] * ac[i - j]
        sum_k +=ac[i+1]
        r=-sum_k/e
        a[i]=r
        # Levinson update: a_new[j] = a[j] + k * a[i-1-j]
        # Must not use a[i-1::-1] when i==0 (becomes a[-1::-1], whole array)
        for j in range((i + 1) // 2):
            tmp1 = a[j]
            tmp2 = a[i - 1 - j]
            a[j] = tmp1 + r * tmp2
            a[i - 1 - j] = tmp2 + r * tmp1
        e *= 1.0 - r * r
        if e <= 1e-3 * ac[0]:
            e = 1e-3 * ac[0]
            break
    lpc[:p] = a
    return e

def celf_fir(x,num,_y,N,ord_):
    rnum = np.zeros(ord_, dtype=np.float32)
    for i in range(ord_):
        rnum[i] = num[ord_ - i - 1]
    i = 0
    while i < N - 3:
        sum4 = np.zeros(4, dtype=np.float32)
        sum4[0]=x[i]
        sum4[1]=x[i+1]
        sum4[2]=x[i+2]
        sum4[3]=x[i+3]
        xcorr_kernel(rnum, x[i - ord_:], sum4, ord_)
        _y[i] =  sum4[0]
        _y[i + 1] =  sum4[1]
        _y[i + 2] =  sum4[2]
        _y[i + 3] =  sum4[3]
        i += 4
    while i < N:
        s = 0.0
        for j in range(ord_):
            s += rnum[j] * x[i + j - ord_]
        _y[i] = x[i] + s
        i += 1

def celf_iir(x,den,_y,N,ord_,mem):
    rden = np.zeros(ord_, dtype=np.float32)
    y=np.zeros(ord_+N,dtype=np.float32)
    for i in range(ord_):
        rden[i] = den[ord_ - i - 1]
        y[i] = -mem[ord_-i-1]
    i = 0
    while i < N - 3:
        sum4 = np.zeros(4, dtype=np.float32)
        sum4[0]=x[i]
        sum4[1]=x[i+1]
        sum4[2]=x[i+2]
        sum4[3]=x[i+3]
        xcorr_kernel(rden, y[i:], sum4, ord_)
        y[i]=-sum4[0]
        _y[i] = sum4[0]
        sum4[1] +=y[i+ord_]*den[0]
        y[i+ord_+1]=-sum4[1] 
        _y[i+1] = sum4[1]
        sum4[2] +=y[i+ord_+1]*den[0]
        sum4[2] +=y[i+ord_]*den[1]
        y[i+ord_+2]=-sum4[2] 
        _y[i+2] = sum4[2]
        sum4[3] +=y[i+ord_+2]*den[0]
        sum4[3] +=y[i+ord_+1]*den[1]
        sum4[3] +=y[i+ord_]*den[2]
        y[i+ord_+3]=-sum4[3] 
        _y[i+3] = sum4[3]
        i += 4
    while i < N:
        s =x[i]
        for j in range(ord_):
            s -= y[i+j]*rden[j]
        y[i+ord_]=s
        _y[i] = s
        i += 1
    for i in range(ord_):
        mem[i] = y[N-i+1]
   
def _celt_autocorr(x,ac,window,overlap,lag,n):
    fastN=n-lag
    xx=np.zeros(n,dtype=np.float32)
    if(overlap==0):
        xptr=x
    else:
        for i in range(n):
            xx[i]=x[i]*x[i]
        for i in range(overlap):
            xx[i]=x[i]*window[i]
            xx[n-i-1]=x[n-i-1]*window[i]
        xptr=xx
    shift_=0
    celt_pitch_xcorr(xptr,xptr,ac,fastN,lag+1)
    for k in range(lag):
        d=0
        for i in range(k+fastN,n):
            d +=xptr[i]*xptr[i-k]
        ac +=d

