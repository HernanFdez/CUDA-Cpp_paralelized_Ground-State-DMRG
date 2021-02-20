#include <stdio.h>
#include<stdlib.h>
#include<assert.h>

#include <cublas.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <cuda_runtime.h>

#include "gpu_timer.h"
#include "cpu_timer.h"

#include <fstream>
#include <iostream>

#include <cusolverDn.h>
#include <time.h>


using namespace std;




#define BLOCK_SIZE 1


double tb_exact(int L){
    double e=0, k, ek;
    for(int a=0;a<L;a++){
        k=M_PI*(a+1)/(L+1);
        ek=-2*cos(k);
        if(ek<0){
            e+=ek;
        }
    }
    return(e);
}


void EigenSolve(int m, double *d_A, double *d_W)
{
    cudaError_t cudaStat1, cudaStat3;
    cusolverStatus_t cusolver_status;

    cusolverDnHandle_t cusolverH = NULL;
    cusolver_status = CUSOLVER_STATUS_SUCCESS;

    int lwork    = 0, 
        info_gpu = 0,
       *devInfo  = NULL;

    double *d_work = NULL;

    cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat3);

    
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);


    int lda=m;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnDsyevd_bufferSize(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_W,
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);


    cusolver_status = cusolverDnDsyevd(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_W,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

    assert(0 == info_gpu);


    if (devInfo  ) cudaFree(devInfo);
    if (d_work   ) cudaFree(d_work );
    if (cusolverH) cusolverDnDestroy(cusolverH);
}





void print_d_vec(int len, double *d_v)
{
    double *v=(double *)malloc(len*sizeof(double));

    cudaMemcpy(v, d_v, len*sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < len; ++i) printf("%lf\n", v[i]); printf("\n");

    free(v);
}

void get_row(int height, int width, int row, double *A, double *g_row)
{
    cublasStatus_t stat;
    cublasHandle_t manija;
    stat=cublasCreate(&manija);

    stat = cublasDcopy(manija, width, A+row, height, g_row, 1);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! Dcopy error\n");
        exit(1);
    }    
}

void transpose(int height, int width, double *A, double *B)
{
    double const alpha(1.0);
    double const beta(0.0);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasDgeam(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N, 
                height, width, 
                &alpha, (const double*)A, width , 
                &beta , (const double*)A, height, 
                                       B, height);
    
    cublasDestroy(handle);
}



void swap(double ** p1, double **p2)
{
	double *swapper=*p1;
	*p1=*p2;
	*p2=swapper;
}



__global__ void SetI_ker(double *d_ptr, int m)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    d_ptr[col*m+row] = (row==col)?1:0;
}

void setI(double *d_ptr, int m)
{
    int size=min(BLOCK_SIZE, m );
    dim3 dimBlock(size, size);
    dim3 dimGrid(m / size, m / size);

    SetI_ker<<<dimGrid,dimBlock>>>(d_ptr,m);
    cudaDeviceSynchronize();
}



void Dgemm_ptr(int N, int K, int M, double *alpha, double *beta, double *A, double *B, double *C)
{
    cublasStatus_t stat;
    cublasHandle_t manija;
    stat=cublasCreate(&manija);

    stat = cublasDgemm(manija,CUBLAS_OP_N,CUBLAS_OP_N, N,M,K, alpha, (const double*)A,N, (const double*)B,K, beta, C,N);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! Dgemm error\n");
        exit(1);
    }
}

void Dgemm(int N, int K, int M, double alpha, double beta, double *A, double *B, double *C)
{
	double al=alpha, bet=beta;
	Dgemm_ptr(N, K, M, &al, &bet, A, B, C);	
}


void Daxpy(int len, double alpha, double *A, double *B)
{
    cublasStatus_t stat;
    cublasHandle_t manija;
    stat=cublasCreate(&manija);

    double al = alpha;

    stat = cublasDaxpy(manija, len, &al, A, 1, B, 1);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! Daxpy error\n");
        exit(1);
    }  

}

void Dscal(int len, double alpha, double *v)
{
    cublasStatus_t stat;
    cublasHandle_t manija;
    stat=cublasCreate(&manija);
    const double al=alpha;

    stat = cublasDscal(manija, len, &al, v, 1);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! Daxpy error\n");
        exit(1);
    }    
}

void Dcopy(int len, double *from, double *to)
{
    cublasStatus_t stat;
    cublasHandle_t manija;
    stat=cublasCreate(&manija);

    stat = cublasDcopy(manija, len, from, 1, to, 1);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! Dcopy error\n");
        exit(1);
    }   
}


double norm2(int len, double *A)
{
    cublasStatus_t stat;
    cublasHandle_t manija;
    stat=cublasCreate(&manija);

    double result;
    stat = cublasDnrm2(manija, len, A, 1, &result);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! norm2 error\n");
        exit(1);
    }   

    return result;
}


__global__ void set_vec_ker(int len, int i, double *d_vec)
{
    int j=blockIdx.x * blockDim.x + threadIdx.x;

    if(j!=i && j<len)   d_vec[j]=0;
    if(j==i         )   d_vec[j]=1;    
}

void set_vec(int height, int width, int i, double *A, double *vec)
{
    int lmax=(height>width)?height:width;

    dim3 dimBlock(i+1);
    dim3 dimGrid(1);

    set_vec_ker<<<dimGrid,dimBlock>>>(lmax, i, vec);
    cudaDeviceSynchronize();

    Dcopy(height-i-1, A+(i*height+i+1), vec+(i+1) );  
}

__global__ void cut_ker(int m, double *Qdat)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row<m && col<row) Qdat[col*m+row]=0;
}

void cut(int m, double *d_ptr)
{
    int size=min(BLOCK_SIZE, m );
    dim3 dimBlock(size, size);
    dim3 dimGrid(m / size, m / size);

    cut_ker<<<dimGrid,dimBlock>>>(m,d_ptr);
    cudaDeviceSynchronize();
}

__global__ void cpy_QR_ker(int height, int width, double *v_aux0, double *v_QRdat, double *v_Q, double *v_R)
{
    int lmax=(height>width)?height:width;
    int lmin=(height<width)?height:width;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row<height && col<width) v_Q[col*height+row]=(row <  height && col < lmin  )? v_aux0 [col*lmax  +row]:0;
    if(row<width  && col<width) v_R[col*width +row]=(row <= col    && row < height)? v_QRdat[col*height+row]:0;
}


void cpy_QR(int height, int width, double *QRdat, double *Q_built, double *g_Q, double *g_R)
{
    int hmax=max(height, width);

    int hsize=min(BLOCK_SIZE, hmax  ),
        wsize=min(BLOCK_SIZE, width );

    dim3 dimBlock(        wsize,        hsize);
    dim3 dimGrid (width / wsize, hmax / hsize);

    cpy_QR_ker<<<dimGrid,dimBlock>>>(height, width, Q_built, QRdat, g_Q, g_R);
    cudaDeviceSynchronize();     
}


void build_QR(int height, int width, double *QRdat, double *h_TAU, double *g_Q, double *g_R)
{
    int ltau=(height<width)?height:width;
    int lmax=(height>width)?height:width;

    double *fact, *vec, *aux0, *aux1;

    cudaMalloc((void **)&aux0, lmax*lmax*sizeof(double));
    cudaMalloc((void **)&aux1, lmax*lmax*sizeof(double));

    cudaMalloc((void **)&fact, lmax*lmax*sizeof(double));
    cudaMalloc((void **)&vec ,      lmax*sizeof(double));



    setI(aux0,lmax);
    setI(aux1,lmax);

    double alpha=1.0, beta=0.0;


    for (int i = 0; i < ltau; ++i)
    {
        setI(fact,lmax);
        set_vec(height, width, i, QRdat, vec);

        Dgemm_ptr(lmax, 1, lmax, h_TAU+i, &alpha, vec, vec, fact);

        if(i%2==0)  Dgemm_ptr(lmax,lmax,lmax, &alpha,&beta, aux0,fact, aux1);
        else        Dgemm_ptr(lmax,lmax,lmax, &alpha,&beta, aux1,fact, aux0);
    }  


    cpy_QR(height, width, QRdat, aux0, g_Q, g_R);


    cudaFree(aux0);
    cudaFree(aux1);
    cudaFree(fact);
    cudaFree(vec );
}


void d_QR(int height, int width, double *d_A, double *g_Q, double *g_R) //QR decomposition of a square matrix
{
    int ltau=(height<width)?height:width;

    double **d_Aarray, **d_TauArray, *d_TAU;

    cudaMalloc((void **)&d_Aarray  ,      sizeof(double*) );
    cudaMalloc((void **)&d_TauArray,      sizeof(double*) );
    cudaMalloc((void **)&d_TAU     , ltau*sizeof(double ) ); 

    double *h_TAU = (double*) malloc(ltau*sizeof(double ) );


    cublasHandle_t cublas_handle;
    cublasStatus_t stat;
    cublasCreate(&cublas_handle);



    double *Aarray  [1] = {d_A  };
    double *TauArray[1] = {d_TAU};

    cudaMemcpy(d_Aarray  , Aarray  , sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TauArray, TauArray, sizeof(double*), cudaMemcpyHostToDevice);


    int info=0;
    stat = cublasDgeqrfBatched(cublas_handle, height, width, d_Aarray, height, d_TauArray, &info, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) printf("\n d_QR failed");

    Dscal(ltau, -1, d_TAU);
    cudaMemcpy(h_TAU, d_TAU, ltau*sizeof(double), cudaMemcpyDeviceToHost);


    build_QR(height, width, d_A, h_TAU, g_Q, g_R);


    cudaFree(d_Aarray);
    cudaFree(d_TauArray);
    cudaFree(d_TAU);
    free(h_TAU);
}

double distrib()
{
    int r = rand();

    double frac = (double) r/RAND_MAX;

    return (frac-1);
}




struct MPS
{
	int n, m, d, link;
	bool canonizedQ;
	double **C;

	MPS(int n, int m, int d):n(n), m(m), d(d)
	{
		canonizedQ=false;
		C=(double **)malloc(n*sizeof(double *));

		double *h_aux=(double *)malloc(m*d*m*sizeof(double));

		int n_elems;
		for (int pos = 0; pos < n; ++pos)
		{
			n_elems=(pos==0 || pos==n-1)?(m*d*m):(m*d*m);

			for (int j = 0; j < n_elems; ++j)	h_aux[j] = 2*distrib()/sqrt(n_elems);

			cudaMalloc((void **)(C+pos), n_elems*sizeof(double) );
			cudaMemcpy(C[pos], h_aux   , n_elems*sizeof(double), cudaMemcpyHostToDevice);
		}

		free(h_aux);
	}

    void lQR(int i)
    {
        if(i<0 || i>=n-1) {cout<<"cannot lQR out of bounds"; return;}
        assert( !(i<0 || i>=n-1) );

        int h1=(i  ==0  )?d:(m*d);
        int w2=(i+1==n-1)?d:(d*m);

        double *g_Q, *g_R, *aux;
        cudaMalloc((void **)&g_Q, m*d*m*sizeof(double) );
        cudaMalloc((void **)&g_R, m*d*m*sizeof(double) );
        cudaMalloc((void **)&aux, m*d*m*sizeof(double) );


        d_QR(h1, m, C[i], g_Q, g_R); 
        swap(&g_Q, C+i);



        Dgemm(m,m,w2, 1, 0, g_R, C[i+1], aux);
        swap(C+(i+1), &aux);

        cudaFree(g_Q);
        cudaFree(g_R);
        cudaFree(aux);
    }

    void rQR(int i)
    {
        double *fact;
        cudaMalloc((void **)&fact, m*m*sizeof(double) );

        if(i<=0 || i>n-1) {cout<<"cannot rQR out of bounds"; return;}
        assert( !(i<=0 || i>n-1) );


        int h1=(i-1==0  )?d:(m*d);
        int w2=(i  ==n-1)?d:(d*m);

        double *g_Q, *g_R, *aux;
        cudaMalloc((void **)&g_Q, m*d*m*sizeof(double) );
        cudaMalloc((void **)&g_R, m*d*m*sizeof(double) );
        cudaMalloc((void **)&aux, m*d*m*sizeof(double) );

        
        transpose(w2, m , C[i], g_Q);
        swap(C+i, &g_Q);
        d_QR(w2, m, C[i], g_Q, g_R);  
        transpose(m , w2, g_Q, C[i]);



        transpose(m,m, g_R, fact);
        Dgemm(h1,m,m, 1,0, C[i-1], fact, aux);
        swap(C+(i-1), &aux); 
        

        cudaFree(fact);   
        cudaFree(g_Q);
        cudaFree(g_R);
        cudaFree(aux);
    }

	void lcanon(int sitio)
	{		
		for(int i = 0  ; i < sitio; i++) lQR(i);
	}

    void rcanon(int sitio)
    {       
        for(int i = n-1; i > sitio; i--) rQR(i);
    }

    void canonize(int sitio)
    {
        lcanon(sitio);
        rcanon(sitio);
        canonizedQ=true;
        link=sitio;
    }


    void sweep(bool forwardQ)
    {
        if(!canonizedQ){cout<<"cannot sweep a non-canonized MPS"<<endl; return;}    
        if((forwardQ && link==n-1) || (!forwardQ && link==0) ){cout<<"cannot sweep out of bounds"<<endl; return;}

        assert(canonizedQ);   assert( (forwardQ && link<n-1) || (!forwardQ && link>0) );

        if(forwardQ) {lQR(link); link++;}
        else         {rQR(link); link--;}
    }    

    double project(int *state)
    {
        double *fact, *g_row;
        cudaMalloc((void **)&fact , d*m*sizeof(double) );
        cudaMalloc((void **)&g_row,   m*sizeof(double) );

        Dcopy(d*m, C[0], fact);

        for (int i = 0; i < n-1; ++i)
        {   
            int w2=(i+1==n-1)?d:(d*m);
            
            get_row(d, m, state[i], fact, g_row);
            
            Dgemm(1,m,w2, 1, 0, g_row, C[i+1], fact);
        }   

        double h_aux[1];
        cudaMemcpy(h_aux, fact+state[n-1], sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(fact);
        cudaFree(g_row);

        return h_aux[0];
    }

    double MPSnorm()
    {
        if(!canonizedQ){cout<<"cannot calculate the norm of a non-canonized MPS"<<endl; return(0);} 

        int n_elems=(link==0 || link==n-1)?(d*m):(m*d*m);
        return norm2(n_elems, C[link]);
    }

    void normalize()
    {
        if(!canonizedQ){cout<<"cannot normalize a non-canonized MPS"<<endl; return;}

        double nrm=MPSnorm();

        int n_elems=(link==0 || link==n-1)?(d*m):(m*d*m);
        Dscal(n_elems, 1/nrm, C[link]);
    }

    void destroy()
    {
        for(int i=0; i<n; i++) cudaFree(C[i]);

        cudaFree(C);
    }

};


double overlap(MPS v1, MPS v2)
{
    if(v1.n!=v2.n || v1.d!=v2.d){cout<<"cannot calculate overlap between different-shaped MPSs"<<endl; return 0;}

    int n=v1.n, d=v1.d, m1=v1.m, m2=v2.m;
    int m1L, m1R, m2L, m2R;

    double *aux0, *aux1;
    cudaMalloc((void**)&aux0, m1*d*m2*sizeof(double) );
    cudaMalloc((void**)&aux1, m1*d*m2*sizeof(double) );



    double h_aux[1]={1};
    cudaMemcpy(aux0, h_aux, sizeof(double), cudaMemcpyHostToDevice);

    for (int i = 0; i < n; ++i)
    {
        if(i==0  ) {m1L=1; m2L=1;}  else {m1L=m1; m2L=m2;}
        if(i==n-1) {m1R=1; m2R=1;}  else {m1R=m1; m2R=m2;}

        transpose(m2L,  m1L, aux0, aux1);  

        Dgemm(m2L, m1L, d*m1R, 1,0, aux1, v1.C[i], aux0);

        transpose(m1R, m2L*d, aux0, aux1);  
  
        Dgemm(m1R, m2L*d, m2R, 1,0, aux1, v2.C[i], aux0);
    }

    cudaMemcpy(h_aux, aux0, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(aux0);
    cudaFree(aux1);

    return h_aux[0];  
}

void left_contract(int end, int m1, int m2, int m_op, int d, double *s1, double *s2, double *s_op, double *from, double *to)
{
    double *aux0, *aux1;
    cudaMalloc((void**)&aux0, m1*d*m2*m_op*sizeof(double) );
    cudaMalloc((void**)&aux1, m1*d*m2*m_op*sizeof(double) );


    int m1L, m1R, m2L, m2R, wL, wR;
    if(end==-1) {m1L=1; wL=1; m2L=1;}  else {m1L=m1; wL=m_op; m2L=m2;}
    if(end== 1) {m1R=1; wR=1; m2R=1;}  else {m1R=m1; wR=m_op; m2R=m2;}


    Dgemm(m2L* wL,  m1L, d*m1R, 1,0, from, s1, aux1);

    transpose(m1R, m2L* wL*d, aux1, aux0);  

    Dgemm(m1R*m2L,  wL*d, d*wR, 1,0, aux0, s_op, aux1);

    transpose( wR, m1R*m2L*d, aux1, aux0);

    Dgemm( wR*m1R, m2L*d,  m2R, 1,0, aux0, s2, aux1);

    transpose(m2R,  wR*  m1R, aux1, to);

    cudaFree(aux0);
    cudaFree(aux1);
}

void right_contract(int end, int m1, int m2, int m_op, int d, double *s1, double *s2, double *s_op, double *from, double *to)
{
    double *aux0, *aux1;
    cudaMalloc((void**)&aux0, m1*d*m2*m_op*sizeof(double) );
    cudaMalloc((void**)&aux1, m1*d*m2*m_op*sizeof(double) );


    int m1L, m1R, m2L, m2R, wL, wR;
    if(end==-1) {m1L=1; wL=1; m2L=1;}  else {m1L=m1; wL=m_op; m2L=m2;}
    if(end== 1) {m1R=1; wR=1; m2R=1;}  else {m1R=m1; wR=m_op; m2R=m2;}


    Dgemm(m2L*d, m2R, wR*m1R, 1,0, s2, from, aux1);

    transpose(d*wR*m1R, m2L, aux1, aux0);  

    Dgemm(wL*d, d*wR, m1R*m2L, 1,0, s_op, aux0, aux1);

    transpose( d*m1R*m2L, wL, aux1, aux0);

    Dgemm( m1L, d*m1R, m2L*wL, 1,0, s1, aux0, aux1);

    transpose(m2L*wL, m1L, aux1, to);

    cudaFree(aux0);
    cudaFree(aux1);
}

void apply(int end, int m1, int m2, int m_op, int d, double *s1, double *s_out, double *s_op, double *left, double *right)
{
    double *aux0, *aux1;
    cudaMalloc((void**)&aux0, m1*d*m2*m_op*sizeof(double) );
    cudaMalloc((void**)&aux1, m1*d*m2*m_op*sizeof(double) );


    int m1L, m1R, m2L, m2R, wL, wR;
    if(end==-1) {m1L=1; wL=1; m2L=1;}  else {m1L=m1; wL=m_op; m2L=m2;}
    if(end== 1) {m1R=1; wR=1; m2R=1;}  else {m1R=m1; wR=m_op; m2R=m2;}

    //cout<<"Apply"<<endl; //print_d_vec(m1L*d*m1R, s1); //print_d_vec(m2L*wL*m1L, left);

    //printf("%d\t%d\t%d\n\n", m2L, wL, m_op);

    Dgemm(m2L* wL,  m1L, d*m1R, 1,0, left, s1, aux1);   //print_d_vec(m2L* wL*d*m1R, aux1);

    transpose(m1R, m2L* wL*d, aux1, aux0);  

    Dgemm(m1R*m2L,  wL*d, d*wR, 1,0, aux0, s_op, aux1);

    transpose(m2L*d*wR, m1R, aux1, aux0);

    transpose(wR*m1R, m2R, right, aux1);

    Dgemm( m2L*d, wR*m1R,  m2R, 1,0, aux0, aux1, s_out);


    


    cudaFree(aux0);
    cudaFree(aux1);
}



double final_contract(int end, int m1, int m2, int m_op, int d, double *s1, double *s2, double *s_op, double *left, double *right)
{
    int dim_s2=(end==0)?(m2*d*m2):(m2*d);

    double *aux_s;
    cudaMalloc((void **)&aux_s, dim_s2*sizeof(double));

    apply(end, m1, m2, m_op, d, s1, aux_s, s_op, left, right);

    double *d_val;
    cudaMalloc((void **)&d_val, sizeof(double) );

    
    Dgemm(1,dim_s2,1, 1,0, aux_s, s2, d_val);

    double h_val;
    cudaMemcpy(&h_val, d_val, sizeof(double), cudaMemcpyDeviceToHost);   

    cudaFree(d_val);

    return   h_val;
}

double sandwich_v0(MPS v1, MPS v2, MPS op)
{
    assert(v1.n!=op.n || v2.n!=op.n || v1.d!=sqrt(op.d) || v2.d!=sqrt(op.d) );

    int n=v1.n, d=v1.d;
    int m1=v1.m, m2=v2.m, m_op=op.m;

    double *aux0;
    cudaMalloc((void**)&aux0, m1*m_op*m2*sizeof(double) );


    double h_aux[1]={1};
    cudaMemcpy(aux0, h_aux, sizeof(double), cudaMemcpyHostToDevice);

    for (int i = 0; i < n; ++i)
    {
        int end=0;
        if(i==0  ) end=-1; 
        if(i==n-1) end= 1;

        left_contract(end, m1, m2, m_op, d, v1.C[i], v2.C[i], op.C[i], aux0, aux0);
    }

    cudaMemcpy(h_aux, aux0, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(aux0);

    return h_aux[0];  
}

double sandwich(MPS v1, MPS v2, MPS op)
{
    if(v1.n!=op.n || v2.n!=op.n || v1.d!=sqrt(op.d) || v2.d!=sqrt(op.d) ){cout<<"cannot sandwich different-shaped MPSs/MPO"<<endl; return 0;}

    int n=v1.n, d=v1.d;
    int m1=v1.m, m2=v2.m, m_op=op.m;


    double h_aux[1]={1};

    double *left, *right, *aux_s;
    cudaMalloc((void **)&left , m1*m_op*m2*sizeof(double) );
    cudaMalloc((void **)&right, m1*m_op*m2*sizeof(double) );
    cudaMalloc((void **)&aux_s, m2*d   *m2*sizeof(double) );

    cudaMemcpy(left , h_aux, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(right, h_aux, sizeof(double), cudaMemcpyHostToDevice);


    int end, pos=v1.canonizedQ? v1.link : 0;

    for (int i = 0; i<pos; i++)
    {
        end=0;
        if(i==0  ) end=-1; 
        if(i==n-1) end= 1;

        left_contract (end, m1, m2, m_op, d, v1.C[i], v2.C[i], op.C[i], left , left );
    }

    for (int i = n-1; i>pos; i--)
    {
        end=0;
        if(i==0  ) end=-1; 
        if(i==n-1) end= 1;

        right_contract(end, m1, m2, m_op, d, v1.C[i], v2.C[i], op.C[i], right, right);
    }

    end=0;
    if(pos==0  ) end=-1; 
    if(pos==n-1) end= 1;


    //print_d_vec(m1*m_op*m2, right);
        
    apply(end, m1, m2, m_op, d, v1.C[pos], aux_s, op.C[pos], left, right);

    double *val;
    cudaMalloc((void **)&val, sizeof(double) );

    Dgemm(1,( (end==-1)?1:m2 )*d*( (end==1)?1:m2 ),1, 1,0, aux_s, v2.C[pos], val);

    cudaMemcpy(h_aux, val, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(val);

    cudaFree(left); 
    cudaFree(right); 
    cudaFree(aux_s);

    return h_aux[0];  
}


__global__ void build_tridiag_ker(int m, double *alpha, double *beta, double *M)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    M[col*m+row] = (row==col)? (alpha[col]):( (col==row+1 || col==row-1)? (beta[min(col,row)]):0 );    
}

void build_tridiag(int m, double *alpha, double *beta, double *M)
{
    dim3 dimBlock(m, m);
    dim3 dimGrid(1, 1);

    build_tridiag_ker<<<dimGrid,dimBlock>>>(m, alpha, beta, M);
    cudaDeviceSynchronize();
}



void Lanczos(int end, int m, int m_op, int d, double *s_eff, double *h_eff, double *left, double *right)
{
    double epsilum=0.001; 
    int K_dim=3;
    int max_iter=100;

    int dim = (end==0)?(m*d*m):(m*d);


    double  *T, *v, *eval, *a, *b;

    cudaMalloc((void **)&T   ,  K_dim* K_dim*sizeof(double) );
    cudaMalloc((void **)&eval,         K_dim*sizeof(double) );

    cudaMalloc((void **)&   v, (K_dim+1)*dim*sizeof(double) );
    cudaMalloc((void **)&   a,  K_dim       *sizeof(double) );
    cudaMalloc((void **)&   b,  K_dim       *sizeof(double) );





    double E = final_contract(end, m, m, m_op, d, s_eff, s_eff, h_eff, left, right);   

    bool first=true;
    for(int iter=0; iter<max_iter; iter++)
    {

        cudaMemcpy(v+0, s_eff, dim*sizeof(double), cudaMemcpyDeviceToDevice);


        for (int i = 0; i < K_dim; ++i)
        {

            apply(end, m, m, m_op, d, v+i*dim, v+(i+1)*dim, h_eff, left, right);

            Dgemm(1,dim,1, 1,0, v+(i+1)*dim, v+i*dim, a+i);


            if(i==K_dim-1) break;

            double h_aux;
            cudaMemcpy(&h_aux, a+i, sizeof(double), cudaMemcpyDeviceToHost);
            Daxpy(dim, -h_aux, v+i*dim, v+(i+1)*dim );

            if(i>0) 
            {
                cudaMemcpy(&h_aux, b+(i-1), sizeof(double), cudaMemcpyDeviceToHost);
                Daxpy(dim, -h_aux, v+(i-1)*dim, v+(i+1)*dim );
            }

            double beta = norm2(dim, v+(i+1)*dim);

            cudaMemcpy(b+i, &beta, sizeof(double), cudaMemcpyHostToDevice);

            Dscal(dim, 1/beta, v+(i+1)*dim );
        }

        build_tridiag(K_dim, a, b, T);

        EigenSolve(K_dim, T, eval);


        double min_val;
        cudaMemcpy(&min_val, eval+0, sizeof(double), cudaMemcpyDeviceToHost);

        Dgemm(dim, K_dim, 1, 1,0, v, T+0, s_eff);

        //cout<<min_val<<"\t"<<"min_val"<<endl;


        if(!first && abs(E- min_val) < epsilum ) break;
        E=min_val; 

        first=false;
    }
    //cout<<endl;

    Dscal(dim, 1/norm2(dim, s_eff), s_eff );   


    cudaFree(T);
    cudaFree(v); 
    cudaFree(eval);
    cudaFree(a);
    cudaFree(b);    
}









void MPS_scale(MPS S, double k)
{
    int n=S.n, m=S.m, d=S.d, link=S.link;

    if(S.canonizedQ)            Dscal((link==0 || link==n-1)?(d*m):(m*d*m), k, S.C[link]);
    else 
    {
        k=pow(k, (double)1/n );
        for(int i=0; i<n; i++)  Dscal((   i==0 ||    i==n-1)?(d*m):(m*d*m), k, S.C[i]   );
    }
}

__global__ void site_add_ker(int end, int m1, int m2, int d, double *s1, double *s2, double *s3)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.z * blockDim.z + threadIdx.z;


    int idx3 = i                    + ((end==-1)?1:(m1+m2)) * (o+d* j                  ),
        idx1 = i                    + ((end==-1)?1: m1    ) * (o+d* j                  ),
        idx2 =(i-((end==-1)?0:m1) ) + ((end==-1)?1: m2    ) * (o+d*(j-((end==1)?0:m1) ));

    s3[idx3]=0;

    if( (i< m1 || end==-1) && (j< m1 || end==1) ) s3[idx3] = s1[idx1];
    if( (i>=m1 || end==-1) && (j>=m1 || end==1) ) s3[idx3] = s2[idx2];  
}

void site_add(int end, int m1, int m2, int d, double *s1, double *s2, double *s3)
{
    int height=(end==-1)?1:(m1+m2), 
        width =(end== 1)?1:(m1+m2);
    
    dim3 dimBlock(width, height, d);
    dim3 dimGrid(1);

    site_add_ker<<<dimGrid,dimBlock>>>(end, m1, m2, d, s1, s2, s3);
    cudaDeviceSynchronize();     
}

MPS MPS_add(MPS v1, MPS v2, bool destroyQ=true)
{
    int n=v1.n, d=v1.d, m1=v1.m, m2=v2.m;   
    MPS sum(n, m1+m2, d);

    assert(v1.n==v2.n);
    assert(v1.d==v2.d);

    for(int site=0; site<n; site++)
    {
        int           end= 0;
        if(site==0  ) end=-1;
        if(site==n-1) end= 1;

        site_add(end, m1, m2, d, v1.C[site], v2.C[site], sum.C[site]);
    } 

    if(destroyQ) {v1.destroy(); v2.destroy(); }

    return sum;  
}

__global__ void site_mult_ker(int end, int m1, int m2, int d, double *s1, double *s2, double *s3)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.z * blockDim.z + threadIdx.z;

    int o1=o%d, o2=o/d;

    int idx1, idx2, idx3;

    int m1L=m1, m2L=m2,
        m1R=m1;

    if(end==-1) {m1L=1; m2L=1;}
    if(end== 1)  m1R=1; 

    double sum=0;

    for(int oo=0; oo<d; oo++)
    {
        idx1 = (i%m1L) + (m1L    ) * ( (o1+d*oo)+(d*d)*(j%m1R) );
        idx2 = (i/m1L) + (    m2L) * ( (oo+d*o2)+(d*d)*(j/m1R) );

        sum += s1[idx1]*s2[idx2];

        //printf("%d\t%d\t%d\t%lf\t%lf\n", idx1, idx2, idx3, s1[idx1], s2[idx2] );

    }
        idx3 = (i    ) + (m1L*m2L) * ( (o      )+(d*d)*(j    ) );

    s3[idx3] = sum;

}

void site_mult(int end, int m1, int m2, int d, double *s1, double *s2, double *s3)
{
    int height=(end==-1)?1:(m1*m2), 
        width =(end== 1)?1:(m1*m2);

    //printf("\n%d\t%d\t%d\n\n", m1, m2, d);
    
    dim3 dimBlock(width, height, d*d);
    dim3 dimGrid(1);

    site_mult_ker<<<dimGrid,dimBlock>>>(end, m1, m2, d, s1, s2, s3);
    cudaDeviceSynchronize();     
}

MPS MPO_multiply(MPS H1, MPS H2, bool destroyQ=true)
{
    int n=H1.n, d=sqrt(H1.d), m1=H1.m, m2=H2.m;
    MPS mult(n, m1*m2, d*d);

    if(H1.n!=H2.n || H1.d!=H2.d) cout<<"cannot multiply different-shaped MPOs"<<endl; 

    else    for(int site=0; site<n; site++)
    {
        int end=0;
        if(site==0  ) end=-1;
        if(site==n-1) end= 1;

        site_mult(end, m1, m2, d, H1.C[site], H2.C[site], mult.C[site]);
    } 

    if(destroyQ) {H1.destroy(); H2.destroy(); }

    return mult;
}



MPS I_op(int n, int d)
{
    MPS O(n, 1, d*d);
    O.canonizedQ=true;
    O.link=0;

    double h_aux[4]={1,0,0,1};

    for (int i = 0; i < n; i++)   cudaMemcpy(O.C[i], h_aux, 4*sizeof(double), cudaMemcpyHostToDevice);

    return(O);
}

MPS Ci(int n, int pos, bool plusQ)
{
    int o=(plusQ ? 2 : 1);

    MPS O=I_op(n,2);

    double h_aux[4]={0,0,0,0}; h_aux[o]=1;
    
    cudaMemcpy(O.C[pos], h_aux, 4*sizeof(double), cudaMemcpyHostToDevice);

    double h_sg[4]={-1,0,0,1};

    for(int i=0; i<pos; i++) cudaMemcpy(O.C[i], h_sg, 4*sizeof(double), cudaMemcpyHostToDevice);

    return O;
}


MPS Ni(int n, int i, bool plusQ)
{
    MPS O=I_op(n,2);

    int o = plusQ?0:3; 

    double h_aux[4]={0,0,0,0}; h_aux[o]=1;

    cudaMemcpy(O.C[i], h_aux, 4*sizeof(double), cudaMemcpyHostToDevice);

    return O;
}


MPS hopp(int n, int s1, int s2)
{
    MPS h1=MPO_multiply(Ci(n,s1,true), Ci(n,s2,false)), 
        h2=MPO_multiply(Ci(n,s2,true), Ci(n,s1,false));

    return MPS_add(h1,h2);
}




MPS TB(int n)
{
    MPS H=hopp(n,0,1);

    for(int i=1; i<n-1; i++) H = MPS_add(H, hopp(n,i,i+1) );
    
    return H;
}


MPS DC(int L, double t1, double t2, double U, double V)
{
    int n=4*L;
    MPS N0=Ni(n,0, true), N1=Ni(n,1,true), N2=Ni(n,2,true), N3=Ni(n,3,true);

    MPS H01=MPO_multiply(N0,N1);
    MPS H23=MPO_multiply(N2,N3); 
    MPS Hi=MPS_add(H01,H23);
    MPS_scale(Hi,U);
    MPS H=Hi;       

    H01=MPS_add(N0,N1);
    H23=MPS_add(N2,N3);
    Hi=MPO_multiply(H01,H23); 
    MPS_scale(Hi,V);
    H=MPS_add(H,Hi);    

    for(int i=1;i<L;i++) 
    {
        for(int j=0;j<4;j++) {Hi=hopp(n,4*(i-1)+j,4*i+j); H=MPS_add(H,Hi);}

        N0=Ni(n,4*i,true); N1=Ni(n,4*i+1,true); N2=Ni(n,4*i+2,true); N3=Ni(n,4*i+3,true);

        H01=MPO_multiply(N0,N1);
        H23=MPO_multiply(N2,N3); 
        Hi=MPS_add(H01,H23);
        MPS_scale(Hi,U);
        H=MPS_add(H,Hi);

        H01=MPS_add(N0,N1);
        H23=MPS_add(N2,N3);
        Hi=MPO_multiply(H01,H23); 
        MPS_scale(Hi,V);
        H=MPS_add(H,Hi);
    }

    return(H);
}




double DMRG(MPS H, int m, int nloops)
{
    int n=H.n, d=sqrt(H.d), m_op=H.m;

    H.canonize(0);

    MPS S(n, m, d); 
    S.canonize(0); 
    S.normalize();




    double **L, **R;
    L=(double **)malloc(n*sizeof(double *));
    R=(double **)malloc(n*sizeof(double *));

    for (int i = 0; i < n; ++i)
    {
        cudaMalloc( (void **)(L+i), ((i==0  )?1:(m*m_op*m))*sizeof(double) );
        cudaMalloc( (void **)(R+i), ((i==n-1)?1:(m*m_op*m))*sizeof(double) );
    }
    double h_aux = 1;
    cudaMemcpy(L[0]  , &h_aux, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(R[n-1], &h_aux, sizeof(double), cudaMemcpyHostToDevice);


    for (int i = n-1; i>0; i--)
    {
        int end=0;
        if(i==0  ) end=-1; 
        if(i==n-1) end= 1;

        right_contract(end, m, m, m_op, d, S.C[i], S.C[i], H.C[i], R[i], R[i-1]);
    }

     

    
    for(int lap=0; lap<nloops; lap++)
    {
        if(lap%1==0) cout<<"############################## "<<"[sweep = "<<lap<<"]"<<endl; 
        
        for(int i=  0; i<n-1; i++)  
        {   
            int end = (i  ==0)?-1:0; 

            cout<<"(sitio = "<<i<<")\t";

            cout<<final_contract(end, m, m, m_op, d, S.C[i], S.C[i], H.C[i], L[i], R[i]) <<endl;

                         Lanczos(end, m,    m_op, d, S.C[i],         H.C[i], L[i], R[i]); 

            S.sweep(true); //H.sweep(true);

            left_contract ( end, m, m, m_op, d, S.C[i], S.C[i], H.C[i], L[i], L[i+1] );
        }

        for(int i=n-1; i>  0; i--)
        {
            int end = (i==n-1)?1:0; 

            cout<<"(sitio = "<<i<<")\t";

            cout<<final_contract(end, m, m, m_op, d, S.C[i], S.C[i], H.C[i], L[i], R[i]) <<endl;

                         Lanczos(end, m,    m_op, d, S.C[i],         H.C[i], L[i], R[i]); 

            S.sweep(false); //H.sweep(false);

            right_contract( end, m, m, m_op, d, S.C[i], S.C[i], H.C[i], R[i], R[i-1] );
        }
    }

    double Ener=sandwich(S, S, H);

    S.destroy();

    return(Ener);
}




int main(int argc, char const *argv[])
{
    srand(0);


    int L=4;

    double t1=1, t2=1, U=0, V=0;
    
    int mMPS=10, nsweeps=10;

    if(argc==2) mMPS= atof(argv[1]);



    MPS H=TB(L);
    //MPS H=DC(L, t1, t2, U, V);



    DMRG(H, mMPS, nsweeps);

    cout<<endl<<"TB_exact = \t"<<tb_exact(L)<<endl;




    return 0;
}

