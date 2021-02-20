#include <iostream>
#include <armadillo>
#include <vector>
#include <iomanip>

#include<assert.h>

#include <cusolverDn.h>

using namespace std;
using namespace arma;


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



void Dgemm(int N, int K, int M, double *alpha, double *beta, double *A, double *B, double *C)
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
	Dgemm(N, K, M, &al, &bet, A, B, C);	
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

void apply(int end, int m1, int m2, int m_op, int d, double *s1, double *s_out, double *s_op, double *left, double *right)
{
    double *aux0, *aux1;
    cudaMalloc((void**)&aux0, m1*d*m2*m_op*sizeof(double) );
    cudaMalloc((void**)&aux1, m1*d*m2*m_op*sizeof(double) );


    int m1L, m1R, m2L, m2R, wL, wR;
    if(end==-1) {m1L=1; wL=1; m2L=1;}  else {m1L=m1; wL=m_op; m2L=m2;}
    if(end== 1) {m1R=1; wR=1; m2R=1;}  else {m1R=m1; wR=m_op; m2R=m2;}


    Dgemm(m2L* wL,  m1L, d*m1R, 1,0, left, s1, aux1);

    transpose(m1R, m2L* wL*d, aux1, aux0);  

    Dgemm(m1R*m2L,  wL*d, d*wR, 1,0, aux0, s_op, aux1);

    transpose(m2L*d*wR, m1R, aux1, aux0);

    transpose(wR*m1R, m2R, right, aux1);

    Dgemm( m2L*d, wR*m1R,  m2R, 1,0, aux0, aux1, s_out);


    cudaFree(aux0);
    cudaFree(aux1);
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







double matnorm(mat &m)
{
	int nrows=m.n_rows, ncols=m.n_cols;
	m.reshape(1,nrows*ncols);
	mat aux(1,1);
	aux=m*m.t();
	m.reshape(nrows,ncols);
	return(sqrt(aux(0,0)));
}

double cubenorm(cube &m)
{
	int nrows=m.n_rows, ncols=m.n_cols, nslices=m.n_slices;
	m.reshape(1,1,nrows*ncols*nslices);
	mat aux(1,1);
	aux=m.tube(0,0);
	aux=aux*aux.t();
	m.reshape(nrows,ncols,nslices);
	return(aux(0,0));
}

void decomp(mat &Q, mat &R, mat &C, int mtrunc=0)			
{
	if(mtrunc>C.n_cols){cout<<"cannot truncate to bigger dimension -> QR decomposition assumed instead"<<endl; mtrunc=0;}
	mat left, right;
	int dim;
		
	if(mtrunc==0)
	{
		qr(left, right, C);
		dim=C.n_cols;
	}
	else
	{
		dim=mtrunc;
		vec ss;
		svd(left, ss, right, C);

		vec vaux=zeros<vec>(dim);
		int nrows=ss.n_rows;
		int copy=min(dim,nrows);
		vaux.head(copy)=ss.head(copy);			
		mat aux=diagmat(vaux);		
		right=right.head_cols(dim);
		right=aux*right.t();				
	}

	Q=zeros<mat>(C.n_rows,dim);  
	R=zeros<mat>(dim,C.n_cols);

	int cols2copy=min(Q.n_cols,left.n_cols);
	int rows2copy=min(R.n_rows,right.n_rows);
	Q.head_cols(cols2copy)=left.head_cols(cols2copy);   
	R.head_rows(rows2copy)=right.head_rows(rows2copy);
}

struct MPS
{
	int n, m, d, link;
	mat s0, sf;
	vector<cube> C;
	bool canonizedQ;

	MPS(int n, int m, int d):n(n), m(m), d(d)
	{
		canonizedQ=false;
		s0=mat(d,m,fill::randn);
		sf=mat(m,d,fill::randn);
		C.resize(n-2);
		for (int i = 0; i < n-2; i++)
		{
			C[i]=cube(m,d,m,fill::randn);
		}
	}

	void lgauge(int sitio, int mtrunc=0)
	{
		if(sitio==0)return;
		if(mtrunc>=m) mtrunc=0;
		int o=(mtrunc!=0) ? mtrunc : m;

		mat Q0(d,o), R(o,m), Q(o*d,o), aux(o,d*m);
		decomp(Q0,R,s0,mtrunc);
		s0=Q0;						
		
		for (int i = 0; i < n-2; i++)
		{
			C[i].reshape(m,d*m,1);
			aux=R*(C[i].slice(0));

			if(i+1==sitio){C[i].slice(0)=aux; C[i].reshape(m,d,m); return;}

			aux.reshape(o*d,m);
			decomp(Q,R,aux,mtrunc);

			C[i]=cube(o*d,o,1);
			C[i].slice(0)=Q;		
			C[i].reshape(o,d,o);
		}
		
		sf=R*sf;
	}

	void truncate(int mtrunc)
	{
		if(mtrunc>=m) cout<<"cannot truncate to bigger dimension"<<endl;
		if(!canonizedQ || link!=0) cout<<"MPS must be 0-canonized for being truncated"<<endl;
		lgauge(n-1,mtrunc);
		m=mtrunc;
		canonize(0);
	}

	void rgauge(int sitio)
	{
		mat Q0(d,m), R(m,m), aux(m,d*m);
		if(sitio==n-1)return;

		mat tr=sf.t();
		decomp(Q0,R,tr);
		sf=Q0.t();
		
		mat Q(m*d,m);
		for (int i = n-3; i >= 0; i--)
		{
			C[i].reshape(m*d,m,1);
			aux=(C[i].slice(0))*R.t();

			if(i+1==sitio){C[i].slice(0)=aux; C[i].reshape(m,d,m); return;}

			aux.reshape(m,d*m);
			aux=aux.t();
			decomp(Q,R,aux);

			C[i].reshape(m,d*m,1);
			C[i].slice(0)=Q.t();
			C[i].reshape(m,d,m);
		}
		
		s0=s0*R.t();
	}

	void canonize(int sitio)
	{
		lgauge(sitio);
		rgauge(sitio);
		canonizedQ=true;
		link=sitio;
	}

	void sweep(bool forwardQ)
	{
		if(!canonizedQ){cout<<"cannot sweep a non-canonized MPS"<<endl; return;}	
		if((forwardQ && link==n-1) || (!forwardQ && link==0) ){cout<<"cannot sweep out bounds"<<endl; return;}
		
		int i=link-1;
		if(forwardQ){link++;}else{link--;}
		if(i==-1){lgauge(1); return;}
		if(i==n-2){rgauge(n-2); return;}
		
		if(forwardQ)
		{
			mat Q(m*d,m), R(m,m), aux(m*d,m);
			C[i].reshape(m*d,m,1);
			aux=C[i].slice(0);

			decomp(Q,R,aux);
			C[i].slice(0)=Q;			
			C[i].reshape(m,d,m);

			if(i==n-3){sf=R*sf; return;}

			C[i+1].reshape(m,d*m,1);
			aux=R*(C[i+1].slice(0));
			C[i+1].slice(0)=aux; 
			C[i+1].reshape(m,d,m);
		}
		else
		{
			mat Q(m*d,m), R(m,m), aux(m,d*m);
			C[i].reshape(m,d*m,1);	
			aux=C[i].slice(0);
			aux=aux.t();		

			decomp(Q,R,aux);
			C[i].slice(0)=Q.t();
			C[i].reshape(m,d,m);	

			if(i==0){s0=s0*R.t(); return;}

			C[i-1].reshape(m*d,m,1);
			aux=(C[i-1].slice(0))*R.t();
			C[i-1].slice(0)=aux; 
			C[i-1].reshape(m,d,m);	
		}
	}

	double project(vector<int> state)
	{
		if(state.size()!=n)cout<<"cannot project into a different-sized state"<<endl;
		int index=state[0];
		mat v(1,m);
		v=(s0.row(index));

		mat aux(m,m);	
		for (int i = 0; i < n-2; ++i)
		{
			index=state[i+1];
			for (int j = 0; j < m; j++)
			{
				for (int k = 0; k < m; k++)
				{
					aux(j,k)=C[i](j,index,k);
				}
			}
	 		v=v*aux;
		} 

		index=state[n-1];
		v=v*(sf.col(index));
		return(v(0,0));
	}

	double MPSnorm()
	{
		if(!canonizedQ){cout<<"cannot calculate the norm of a non-canonized MPS"<<endl; return(0);}	

		if(link==0)return(matnorm(s0));
		if(link==n-1)return(matnorm(sf));

		int index=link-1;
		return(cubenorm(C[index]) );
	}

	void normalize()
	{
		if(!canonizedQ){cout<<"cannot normalize a non-canonized MPS"<<endl; return;}

		double fact=MPSnorm();
		if(link==0)   {s0/=fact; return;}
		if(link==n-1) {sf/=fact; return;}
		C[link-1]/=fact;
	}
};


double firstoverlap(MPS &v1, MPS &v2)
{
	if(!v1.canonizedQ || !v2.canonizedQ){cout<<"cannot overlap a non-canonized MPS"<<endl; return 0;}
	if(v1.link!=v2.link || v1.n!=v2.n || v1.m!=v2.m || v1.d!=v2.d){cout<<"cannot calculate overlap between different-shaped MPSs"<<endl; return 0;}

	
	int link=v1.link, n=v1.n, m=v1.m, d=v1.d;
	mat Q(m,m), R(m,m), aux(m,m);
	double sum;

	Q=(v1.s0).t()*(v2.s0);
	for(int site=0; site<link-1; ++site)
	{
		for(int i=0; i<m; ++i) for(int j=0; j<m; ++j)
		{
			sum=0;
			for(int k=0; k<m; ++k) for(int l=0; l<m; ++l) for(int o=0; o<d; ++o)
			{
				sum+=Q(k,l)*v1.C[site](k,o,i)*v2.C[site](l,o,j);
			}
			aux(i,j)=sum;			
		}
		Q=aux;	
	}

	R=(v1.sf)*(v2.sf).t();
	for(int site=n-3; site>link-1; --site)
	{
		for(int i=0; i<m; ++i) for(int j=0; j<m; ++j)
		{
			sum=0;
			for(int k=0; k<m; ++k) for(int l=0; l<m; ++l) for(int o=0; o<d; ++o)
			{
				sum+=R(k,l)*v1.C[site](i,o,k)*v2.C[site](j,o,l);
			}
			aux(i,j)=sum;			
		}
		R=aux;
	}

	sum=0;
	for(int k=0; k<m; ++k) for(int l=0; l<m; ++l) for(int kk=0; kk<m; ++kk) for(int ll=0; ll<m; ++ll) for(int o=0; o<d; ++o)
	{
		sum+= v1.C[link-1](k,o,kk)*v2.C[link-1](l,o,ll)*Q(k,l)*R(kk,ll);
	}
	return(sum);
}

double firstsandwich(MPS &v1, MPS &v2, MPS &op)
{
	if( !v1.canonizedQ || !v2.canonizedQ || !op.canonizedQ ){cout<<"cannot sandwich non-canonized MPS or MPO"<<endl; return 0;}
	if( (v1.link!=op.link || v2.link!=op.link) || (v1.n!=op.n || v2.n!=op.n) || (v1.m!=op.m || v2.m!=op.m) || (v1.d!=sqrt(op.d) || v2.d!=sqrt(op.d)) ){cout<<"cannot sandwich different-shaped MPS or MPO"<<endl; return 0;}

	int link=v1.link, n=v1.n, m=v1.m, d=v1.d;
	cube Q(m,m,m), R(m,m,m), aux(m,m,m);
	double sum;

	for(int i=0; i<m; ++i) for(int j=0; j<m; ++j) for(int k=0; k<m; ++k)
	{
		sum=0;
		for(int o=0; o<d; ++o) for(int p=0; p<d; ++p)
		{
			sum+=v1.s0(o,i)*op.s0(d*o+p,j)*v2.s0(p,k);
		}
		aux(i,j,k)=sum;			
	}
	Q=aux;
	for(int site=0; site<link-1; ++site)
	{	
		for(int i=0; i<m; ++i) for(int j=0; j<m; ++j) for(int k=0; k<m; ++k)
		{
			sum=0;
			for(int ii=0; ii<m; ++ii) for(int jj=0; jj<m; ++jj) for(int kk=0; kk<m; ++kk) for(int o=0; o<d; ++o) for(int p=0; p<d; ++p)
			{
				sum+=Q(ii,jj,kk)*v1.C[site](ii,o,i)*op.C[site](jj,d*o+p,j)*v2.C[site](kk,p,k);
			}
			aux(i,j,k)=sum;			
		}
		Q=aux;  
		
	}
	
	for(int i=0; i<m; ++i) for(int j=0; j<m; ++j) for(int k=0; k<m; ++k)
	{
		sum=0;
		for(int o=0; o<d; ++o) for(int p=0; p<d; ++p)
		{
			sum+=v1.sf(i,o)*op.sf(j,d*o+p)*v2.sf(k,p);
		}
		aux(i,j,k)=sum;			
	}
	R=aux;
	for(int site=n-3; site>link-1; --site)
	{
		for(int i=0; i<m; ++i) for(int j=0; j<m; ++j) for(int k=0; k<m; ++k)
		{
			sum=0;
			for(int ii=0; ii<m; ++ii) for(int jj=0; jj<m; ++jj) for(int kk=0; kk<m; ++kk) for(int o=0; o<d; ++o) for(int p=0; p<d; ++p)
			{
				sum+=R(ii,jj,kk)*v1.C[site](i,o,ii)*op.C[site](j,d*o+p,jj)*v2.C[site](k,p,kk);  
			}
			aux(i,j,k)=sum;			
		}
		R=aux;
	}

	sum=0;
	for(int i=0; i<m; ++i) for(int j=0; j<m; ++j) for(int k=0; k<m; ++k) for(int ii=0; ii<m; ++ii) for(int jj=0; jj<m; ++jj) for(int kk=0; kk<m; ++kk) for(int o=0; o<d; ++o) for(int p=0; p<d; ++p)
	{
		sum+= Q(i,j,k)*v1.C[link-1](i,o,ii)*op.C[link-1](j,d*o+p,jj)*v2.C[link-1](k,p,kk)*R(ii,jj,kk);
	}
	return(sum);
}


MPS I_op(int n, int d)
{
	MPS O(n,1,d*d);
	O.canonizedQ=true;
	mat aux(d,d,fill::eye);
	aux.reshape(d*d,1);
	O.s0=aux;
	aux.reshape(1,d*d);	
	O.sf=aux;
	O.C.resize(n-2);
	cube cubo(1,d*d,1);
	cubo.slice(0)=aux;  
	for (int i = 0; i < n-2; i++)
	{
		O.C[i]=cubo;
	}
	return(O);
}

void Apply_Heff(const cube &L, const cube &R, const cube &W, const cube &s, cube &z, int m, int m_W, int d, int mm=0)
{
	if(mm==0) mm=m;
	cube E(m*mm,m_W,d), A(m*m,m_W,d);
	double sum;

	/*
	for(int ii=0; ii<m; ii++) for(int k=0; k<mm; k++) for(int jj=0; jj<m_W; jj++) for(int p=0; p<d; p++) 
	{
		sum=0;
		for(int kk=0; kk<mm; kk++) sum+=R(ii,jj,kk)*s(k,p,kk);
		A(mm*ii+k,jj,p)=sum;
	}
	*/

	const mat Rm=mat((double *)R.memptr(),m*m_W,m,false);
	const mat Sm=mat((double *)s.memptr(),m*d,m,false);
	mat Am=Rm*Sm.t();

	for(int ii=0; ii<m; ii++) for(int k=0; k<mm; k++) for(int jj=0; jj<m_W; jj++) for(int p=0; p<d; p++) 
	{
		A(mm*ii+k,jj,p)=Am(ii+m*jj,k+p*m);
	}

	for(int ii=0; ii<m; ii++) for(int k=0; k<mm; k++) for(int j=0; j<m_W; j++) for(int o=0; o<d; o++) 
	{
		sum=0;
		for(int jj=0; jj<m_W; jj++) for(int p=0; p<d; p++) sum+=A(mm*ii+k,jj,p)*W(j,d*o+p,jj);
		E(mm*ii+k,j,o)=sum;
	}

	for(int ii=0; ii<m; ii++) for(int i=0; i<m; i++)for(int o=0; o<d; o++) 
	{
		sum=0;
		for(int k=0; k<m; k++) for(int j=0; j<m_W; j++) sum+=E(mm*ii+k,j,o)*L(i,j,k);
		z(i,o,ii)=sum;
	}
}


void Apply_Heff_ext(cube &supercube, mat &W, mat &s, mat &z, int m, int m_W, int d, bool leftQ)
{
	cube supercube_ext(m,d*d,m);
	double sum;

	for (int i=0; i<m; ++i) for (int k=0; k<m; ++k) for (int o=0; o<d; ++o) for (int p=0; p<d; ++p)
	{
		sum=0;
		for (int j=0; j<m_W; ++j) sum+=(leftQ ? W(d*o+p,j) : W(j,d*o+p) )*supercube(i,j,k);
		supercube_ext(i,d*o+p,k)=sum;
	}

	for (int i=0; i<m; ++i) for (int o=0; o<d; ++o)	
	{
		sum=0; 
		for (int k=0; k<m; ++k) for (int p=0; p<d; ++p) sum+= supercube_ext(i,d*o+p,k)*( leftQ ? s(p,k) : s(k,p) );
		if(leftQ) z(o,i)=sum; else z(i,o)=sum;
	}
}

void contract_0(cube &supercube, mat &s1, mat &s2, mat &W, int m, int m_W, int d, bool leftQ, int mm=0)
{
	if(mm==0) mm=m;
	double sum;
	for(int i=0; i<m; ++i) for(int j=0; j<m_W; ++j) for(int k=0; k<mm; ++k)
	{
		sum=0;
		for(int o=0; o<d; ++o) for(int p=0; p<d; ++p) sum+= (leftQ ? s1(o,i) : s1(i,o)) *(leftQ ? W(d*o+p,j) : W(j,d*o+p))*(leftQ ? s2(p,k) : s2(k,p));
		supercube(i,j,k)=sum;			
	}
}

void contract_step(cube &supercube, cube &newsupercube, cube &s1, cube &s2, cube &W, int m, int m_W, int d, bool leftQ, int mm=0)
{
	if(mm==0) mm=m;
	cube auxL(m*m_W,d,mm), auxC(m*mm,d*d,m_W);
	
	double sum;
	for(int i=0; i<m; ++i) for(int jj=0; jj<m_W; ++jj) for(int kk=0; kk<mm; ++kk) for(int o=0; o<d; ++o)
	{
		sum=0;
		for(int ii=0; ii<m; ++ii) sum+=supercube(ii,jj,kk)*(leftQ ? s1(ii,o,i) : s1(i,o,ii) );
		auxL(m_W*i+jj,o,kk)=sum;			
	}  	
	
	for(int i=0; i<m; ++i) for(int jj=0; jj<m_W; ++jj) for(int k=0; k<mm; ++k) for(int o=0; o<d; ++o) for(int p=0; p<d; ++p)
	{
		sum=0;
		for(int kk=0; kk<mm; ++kk) sum+= auxL(m_W*i+jj,o,kk)*(leftQ ? s2(kk,p,k) : s2(k,p,kk) );
		auxC(mm*i+k,d*o+p,jj)=sum;			
	}  	
	for(int i=0; i<m; ++i) for(int j=0; j<m_W; ++j) for(int k=0; k<mm; ++k)
	{
		sum=0;
		for(int jj=0; jj<m_W; ++jj) for(int o=0; o<d; ++o) for(int p=0; p<d; ++p) sum+= auxC(mm*i+k,d*o+p,jj)*(leftQ ? W(jj,d*o+p,j) : W(j,d*o+p,jj) ) ;
		newsupercube(i,j,k)=sum;
	}
}

double contract_site(cube &s1, cube &s2, int m, int d)
{
	double sum=0;
	for(int i=0; i<m; ++i) for(int ii=0; ii<m; ++ii) for(int o=0; o<d; ++o) sum+= s1(i,o,ii)*s2(i,o,ii);	
	return sum;
}

double sandwich(MPS &v1, MPS &v2, MPS &op)  //doesnt work with end-canonization !
{
	if(  (v1.n!=op.n || v2.n!=op.n) || (v1.d!=sqrt(op.d) || v2.d!=sqrt(op.d)) ){cout<<"cannot sandwich different-shaped MPS or MPO"<<endl; return 0;}

	int link=v1.link, n=v1.n, d=v1.d;
	cube Q(v1.m,op.m,v2.m), R(v1.m,op.m,v2.m), z(v1.m,d,v1.m);

	contract_0(Q, v1.s0, v2.s0, op.s0, v1.m, op.m, d, true, v2.m);		
	contract_0(R, v1.sf, v2.sf, op.sf, v1.m, op.m, d, false, v2.m);	
	for(int site=0; site<link-1; ++site)   contract_step(Q, Q, v1.C[site], v2.C[site], op.C[site], v1.m, op.m, d, true, v2.m);
	for(int site=n-3; site>link-1; --site) contract_step(R, R, v1.C[site], v2.C[site], op.C[site], v1.m, op.m, d, false, v2.m);

	Apply_Heff(Q, R, op.C[link-1], v2.C[link-1], z, v1.m, op.m, d) ;
	return contract_site(v1.C[link-1], z, v1.m, d);
}

double overlap(MPS &v1, MPS &v2)
{
	if(!v1.canonizedQ || !v2.canonizedQ){cout<<"cannot overlap a non-canonized MPS"<<endl; return 0;}
	if(v1.link!=v2.link || v1.n!=v2.n || v1.m!=v2.m || v1.d!=v2.d){cout<<"cannot calculate overlap between different-shaped MPSs"<<endl; return 0;}

	
	int link=v1.link, n=v1.n, m=v1.m, d=v1.d;
	mat Q(m,m), R(m,m);
	cube aux(m,d,m), aux2(m,d,m);
	double sum;

	Q=(v1.s0).t()*(v2.s0);
	for(int site=0; site<link-1; ++site)
	{
		for(int l=0; l<m; ++l) for(int o=0; o<d; ++o) for(int i=0; i<m; ++i) 
		{
			sum=0;
			for(int k=0; k<m; ++k) sum+=Q(k,l)*v1.C[site](k,o,i);
			aux(l,o,i)=sum;
		}
		for(int i=0; i<m; ++i) for(int j=0; j<m; ++j)
		{
			sum=0;
			for(int l=0; l<m; ++l) for(int o=0; o<d; ++o) sum+=aux(l,o,i)*v2.C[site](l,o,j);
			Q(i,j)=sum;			
		}
	}

	R=(v1.sf)*(v2.sf).t();
	for(int site=n-3; site>link-1; --site)
	{
		for(int i=0; i<m; ++i) for(int o=0; o<d; ++o) for(int l=0; l<m; ++l)  
		{
			sum=0;
			for(int k=0; k<m; ++k) sum+=R(k,l)*v1.C[site](i,o,k);
			aux(i,o,l)=sum;			
		}
		for(int i=0; i<m; ++i) for(int j=0; j<m; ++j)
		{
			sum=0;
			for(int l=0; l<m; ++l) for(int o=0; o<d; ++o) sum+=aux(i,o,l)*v2.C[site](j,o,l);
			R(i,j)=sum;			
		}
	}

	for(int l=0; l<m; ++l) for(int o=0; o<d; ++o) for(int kk=0; kk<m; ++kk)
	{
		sum=0;
		for(int k=0; k<m; ++k) sum+= v1.C[link-1](k,o,kk)*Q(k,l);
		aux(l,o,kk)=sum;
	}
	for(int l=0; l<m; ++l) for(int o=0; o<d; ++o) for(int ll=0; ll<m; ++ll)
	{
		sum=0;
		for(int kk=0; kk<m; ++kk) sum+= aux(l,o,kk)*R(kk,ll);
		aux2(l,o,ll)=sum;
	}
	sum=0;
	for(int l=0; l<m; ++l) for(int o=0; o<d; ++o) for(int ll=0; ll<m; ++ll) sum+=aux2(l,o,ll)*v2.C[link-1](l,o,ll);
	return(sum);
}

double MatLanczos(mat M, double epsilum, int d_red)
{
	int dim=M.n_cols;
    mat u(dim,1,fill::randu), evec(d_red,d_red), v(dim, d_red), v_prev, w(dim,1);
    vec eval(d_red), a(d_red), b(d_red-1);
    double E;
    bool first=true;

    while(true)
    {
	    v.col(0)=u/matnorm(u);
	    u=M*v.col(0);
		a(0)=dot(u,v.col(0));
		w=u-a[0]*v.col(0);


		for (int i = 1; i < d_red; ++i)
		{
			b(i-1)=matnorm(w);
			v.col(i)=w/b(i-1);
			u=M*v.col(i);
			a(i)=dot(u,v.col(i));
			w=u-a(i)*v.col(i)-b(i-1)*v.col(i-1);

		}


		mat Mred=diagmat(a)+diagmat(b,1)+diagmat(b,-1);

	    eig_sym(eval,evec,Mred);
	    //cout<<eval(0)<<endl;

	    if(!first && abs(E-eval(0))<epsilum) return(eval(0));

	    E=eval(0); 
	    u=v*evec.col(0);
	    first=false;
    }
}

void d_Lanczos(int end, int m, int m_op, int d, double *s_eff, double *h_eff, double *left, double *right)
{
    double epsilum=0.001; 
    int K_dim=3;
    int max_iter=100;

    int dim=(end==0)?(m*d*m):(m*d);

    double  *T, *v, *eval, *a, *b;

    cudaMalloc((void **)&T, K_dim*K_dim*sizeof(double) );
    cudaMalloc((void **)&eval,    K_dim*sizeof(double) );

    cudaMalloc((void **)&   v, (K_dim+1)*dim*sizeof(double) );
    cudaMalloc((void **)&   a,  K_dim       *sizeof(double) );
    cudaMalloc((void **)&   b,  K_dim       *sizeof(double) );


    double E;
    bool first=true;

    for(int iter=0; iter<max_iter; iter++)
    {
        Dscal(dim, 1/norm2(dim, s_eff), s_eff );

        
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

        Dgemm(dim, K_dim, 1, 1,0, v, T+0, s_eff);

        double min_val;
        cudaMemcpy(&min_val, eval+0, sizeof(double), cudaMemcpyDeviceToHost);



        if(!first && abs(E- min_val) < epsilum ) break;
        E=min_val; 

        first=false;
    }


    cudaFree(T);
    cudaFree(v); 
    cudaFree(eval);
    cudaFree(a);
    cudaFree(b);    
}


void vec_to_array(cube &cubo, double *d_ptr)
{
	int dim=cubo.n_rows;

	double *h_aux;
	h_aux = (double *) malloc(dim*sizeof(double) );

	for(int i=0; i<dim; i++)
	{
		h_aux[i] = cubo(i,0,0);
	}

	cudaMemcpy(d_ptr, h_aux, dim*sizeof(double), cudaMemcpyHostToDevice);
}

void array_to_vec(cube &cubo, double *d_ptr)
{
	int dim=cubo.n_rows;

	double *h_aux;
	h_aux = (double *) malloc(dim*sizeof(double) );

	cudaMemcpy( h_aux, d_ptr, dim*sizeof(double), cudaMemcpyDeviceToHost);

	for(int i=0; i<dim; i++)
	{
		cubo(i,0,0) = h_aux[i] ;
	}

}



void Lanczos_gpu(cube &L, cube &R, cube &H, cube &S, bool ext, bool leftQ=false)
{
	//cout<<"GPU_Lanczos"<<"\t";

	int end = ext? (leftQ? -1:1) :0;

	int m   =(ext &&  leftQ)? S.n_cols : S.n_rows,
	    m_op=(ext &&  leftQ)? H.n_cols : H.n_rows, 
	 	d   =(ext &&  leftQ)? S.n_rows : S.n_cols;


    int mL, mR, wL, wR;
    if(end==-1) {mL=1; wL=1;}  else {mL=m; wL=m_op;}
    if(end== 1) {mR=1; wR=1;}  else {mR=m; wR=m_op;}


	int Sdim = mL*d*mR,
		Hdim = wL*d*d*wR,

		Ldim = mL*wL*mL,
		Rdim = mR*wR*mR;


	cube auxL, auxR, auxH=H, auxS=S;
	auxH.reshape(Hdim, 1, 1);
	auxS.reshape(Sdim, 1, 1);


	double *d_L, *d_R, *d_H, *d_S;
	cudaMalloc((void **)&d_L, Ldim*sizeof(double) );
	cudaMalloc((void **)&d_R, Rdim*sizeof(double) );
	cudaMalloc((void **)&d_H, Hdim*sizeof(double) );
	cudaMalloc((void **)&d_S, Sdim*sizeof(double) );


	double val=1;

	if(ext &&   leftQ) cudaMemcpy(d_L, &val, sizeof(double), cudaMemcpyHostToDevice);
	else			   {auxL=L; auxL.reshape(Ldim, 1, 1); vec_to_array(auxL, d_L);}

	if(ext &&  !leftQ) cudaMemcpy(d_R, &val, sizeof(double), cudaMemcpyHostToDevice);
	else			   {auxR=R; auxR.reshape(Rdim, 1, 1); vec_to_array(auxR, d_R);}

	vec_to_array(auxH, d_H);
	vec_to_array(auxS, d_S);


	d_Lanczos(end, m,  m_op, d, d_S, d_H, d_L, d_R);


	array_to_vec(auxS, d_S);

	auxS.reshape(mL, d, mR);

	S=auxS;
}

void Lanczos_cpu(cube &L, cube &R, cube &H, cube &S, bool ext, bool leftQ=false)
{
	//cout<<"CPU_Lanczos"<<"\t";

	double epsilum=0.001; 
	int d_red=3;
	int max_iter=1000;


	int m= (ext && leftQ) ? S.n_cols : S.n_rows, m_H=(ext && leftQ) ? H.n_cols : H.n_rows, d= S.size()/(ext ? m : m*m), dim=ext ? m*d : m*d*m;

    mat evec(d_red,d_red), v(dim, d_red), u(dim,1), w(dim,1);
    vec eval(d_red), a(d_red), b(d_red-1);

    mat  auxm(m,d  ); if(leftQ){auxm.reshape(d,m);}
    cube auxc(m,d,m);

    double E;
    bool first=true;

	if(ext) auxm=S.slice(0); 		else auxc=S; 						
    if(ext) auxm.reshape(dim,1); 	else auxc.reshape(dim,1,1);
    if(ext) u=auxm; 				else u=auxc.slice(0);					

    for(int iter=0; iter<max_iter; iter++)
    {
    	u/=matnorm(u);
	    v.col(0)=u;

	    if(ext) auxm=u; 															else auxc.slice(0)=u;
	    if(ext) {if(leftQ) auxm.reshape(d,m); else auxm.reshape(m,d);} 				else auxc.reshape(m,d,m);
	    if(ext) Apply_Heff_ext(L, H.slice(0), auxm, auxm, m, m_H, d, leftQ); 		else Apply_Heff(L, R, H, auxc, auxc, m, m_H, d);
	    if(ext) auxm.reshape(dim,1); 												else auxc.reshape(dim,1,1);
	    if(ext) u=auxm; 															else u=auxc.slice(0);

		a(0)=dot(u,v.col(0));
		w=u-a[0]*v.col(0);

		for (int i = 1; i < d_red; ++i)
		{
			b(i-1)=matnorm(w);
			v.col(i)=w/b(i-1);

			if(ext) auxm=v.col(i); 													else auxc.slice(0)=v.col(i);
			if(ext) {if(leftQ) auxm.reshape(d,m); else auxm.reshape(m,d);} 			else auxc.reshape(m,d,m);
		    if(ext) Apply_Heff_ext(L, H.slice(0), auxm, auxm, m, m_H, d, leftQ); 	else Apply_Heff(L, R, H, auxc, auxc, m, m_H, d);
		    if(ext) auxm.reshape(dim,1); 											else auxc.reshape(dim,1,1);
		    if(ext) u=auxm; 														else u=auxc.slice(0);

			a(i)=dot(u,v.col(i));
			w=u-a(i)*v.col(i)-b(i-1)*v.col(i-1);
		}

		mat Mred=diagmat(a)+diagmat(b,1)+diagmat(b,-1);
	    eig_sym(eval,evec,Mred);

	    u=v*evec.col(0);		//cout<<eval(0)<<endl;
	    if(!first && abs(E-eval(0))<epsilum) break;
	    E=eval(0); 

	    first=false;
    }

    if(ext) auxm=u/matnorm(u); 											else auxc.slice(0)=u/matnorm(u);
    if(ext) {if(leftQ) auxm.reshape(d,m); else auxm.reshape(m,d);} 		else auxc.reshape(m,d,m);
    if(ext) S.slice(0)=auxm; 										 	else S=auxc;
}

#define Lanczos Lanczos_cpu 






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

double dot(mat u, mat v)
{
	mat aux=u*v.t();
	return(aux(0,0));
}



MPS MPS_add(MPS &v1, MPS &v2)
{
	assert(v1.n==v2.n && v1.d==v2.d);

	int n=v1.n, d=v1.d, m1=v1.m, m2=v2.m;	
	MPS add(n,m1+m2,d);

	for(int o=0; o<d; o++) 
	{
		for(int i=0; i<m1; i++)  { add.s0(o,i)   =v1.s0(o,i); add.sf(i,o)   =v1.sf(i,o); }
		for(int i=0; i<m2; i++)  { add.s0(o,i+m1)=v2.s0(o,i); add.sf(i+m1,o)=v2.sf(i,o); }
	}

	for(int site=0; site<n-2; site++)
	{
		add.C[site]=zeros<cube>(m1+m2,d,m1+m2);
		for(int o=0; o<d; o++) 
		{
			for(int i=0; i<m1; i++) for(int j=0; j<m1; j++) add.C[site](i,o,j)=v1.C[site](i,o,j);
			for(int i=0; i<m2; i++) for(int j=0; j<m2; j++) add.C[site](i+m1,o,j+m1)= v2.C[site](i,o,j);
		}
	}

	return add;
}

MPS MPO_multiply(MPS &H1, MPS &H2)
{
	if(H1.n!=H2.n || H1.d!=H2.d) cout<<"cannot multiply different-shaped MPOs"<<endl; 
	int n=H1.n, d=sqrt(H1.d), m1=H1.m, m2=H2.m;
	MPS mult(n,m1*m2,d*d);
	double sum, sum2;
	

	for(int o=0; o<d; o++) for(int p=0; p<d; p++) for(int ii=0; ii<m1; ii++) for(int jj=0; jj<m2; jj++)
	{
		sum=0, sum2=0;
		for(int q=0; q<d; q++) {sum+=H1.s0(d*o+q,ii)*H2.s0(d*q+p,jj);  sum2+=H1.sf(ii,d*o+q)*H2.sf(jj,d*q+p);}
		mult.s0(d*o+p,m2*ii+jj)=sum; mult.sf(m2*ii+jj,d*o+p)=sum2; 
	}

	for(int site=0; site<n-2; site++)
	{
		for(int o=0; o<d; o++) for(int p=0; p<d; p++) for(int i=0; i<m1; i++) for(int j=0; j<m2; j++) for(int ii=0; ii<m1; ii++) for(int jj=0; jj<m2; jj++)
		{
			sum=0;
			for(int q=0; q<d; q++) sum+=H1.C[site](i,d*o+q,ii)*H2.C[site](j,d*q+p,jj);
			mult.C[site](m2*i+j,d*o+p,m2*ii+jj)=sum;
		}
	}

	return mult;
}

void MPS_scale(MPS &S, double k)
{
	k=pow(k,(double)1/S.n);
	S.s0*=k;
	S.sf*=k;
	for(int i=0;i<S.n-2;i++) S.C[i]*=k;
}


MPS Ci(int n, int Ni, int plusQ)
{
	int o=(plusQ ? 2 : 1);
	MPS O=I_op(n,2);
	if(Ni==0)  {O.s0=zeros<mat>(4,1);  O.s0(o, 0 )=1; return O;}
	if(Ni==n-1){O.sf=zeros<mat>(1,4);  O.sf(0, o )=1; return O;}
	O.C[Ni-1]=zeros<cube>(1,4,1); O.C[Ni-1](0,o,0)=1; return O;
}

MPS Sgi(int n, int Ni)
{
	MPS O=I_op(n,2);
	if(Ni==0)  {O.s0(0, 0 )=-1; return O;}
	if(Ni==n-1){O.sf(0, 0 )=-1; return O;}
	O.C[Ni-1](0,0,0)=-1; return O;
}

MPS Ni(int n, int Ni)
{
	MPS O=I_op(n,2);
	if(Ni==0)  {O.s0(3,0)=0; return O;}
	if(Ni==n-1){O.sf(0,3)=0; return O;}
	O.C[Ni-1](0,3,0)=0; return O;
}

MPS hopp(int n, int s1, int s2)
{
	MPS h1=I_op(n,2), h2=h1;

	if(s1==0)   
	{
		h1.s0=zeros<mat>(4,1);	h2.s0=zeros<mat>(4,1); 
		h1.s0(2, 0 )=1;			h2.s0(1, 0 )=1;
	} 
	else 
	{
		h1.C[s1-1]=zeros<cube>(1,4,1);	h2.C[s1-1]=zeros<cube>(1,4,1);
		h1.C[s1-1](0,2,0)=1;			h2.C[s1-1](0,1,0)=1;
	}


	if(s2==n-1) 
	{
		h1.sf=zeros<mat>(1,4);	h2.sf=zeros<mat>(1,4);
		h1.sf(0, 1 )=1;			h2.sf(0, 2 )=1;
	} 
	else 
	{
		h1.C[s2-1]=zeros<cube>(1,4,1);	h2.C[s2-1]=zeros<cube>(1,4,1); 
		h1.C[s2-1](0,1,0)=1;			h2.C[s2-1](0,2,0)=1;
	}


	for (int i = s1+1; i < s2; ++i) {h1.C[i-1](0,0,0)=-1; h2.C[i-1](0,0,0)=-1;}

	MPS H=MPS_add(h1,h2);
	return(H);
}


MPS TB(int n)
{
	MPS Hi=hopp(n,0,1), H=Hi;
	for(int i=1; i<n-1; i++) {Hi=hopp(n,i,i+1) ;H=MPS_add(H,Hi);}
	return H;
}


MPS DC0(int L, double t1, double t2, double U, double V)
{
	int n=4*L;
	MPS N0=Ni(n,0), N1=Ni(n,1), N2=Ni(n,2), N3=Ni(n,3);

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

		N0=Ni(n,4*i); N1=Ni(n,4*i+1); N2=Ni(n,4*i+2); N3=Ni(n,4*i+3);

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

MPS DC(int L, double t1, double t2, double U, double V)
{
	int n=4*L;
	MPS N0=Ni(n,0), N1=Ni(n,L), N2=Ni(n,2*L), N3=Ni(n,3*L);

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
		for(int j=0;j<4;j++) {Hi=hopp(n,(i-1)+j*L,i+j*L); MPS_scale(Hi,((j<2) ? t1 : t2)); H=MPS_add(H,Hi);}

		N0=Ni(n,i); N1=Ni(n,i+L); N2=Ni(n,i+2*L); N3=Ni(n,i+3*L);

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

double ground(mat B){
    vec eval;
    mat evec;
    eig_sym(eval,evec,B);
    double minenerg =eval(0);
    return(minenerg);
}

mat I_mat(int l){
	int d=1<<l;
    mat I(d,d,fill::eye);
    return(I);
}

mat N_mat(int n, int Ni)
{
	mat N(2,2,fill::zeros);
	N(0,0)=1;

	mat M=kron(kron(I_mat(Ni),N), I_mat(n-Ni-1));
	return M;
}

mat NiNj(int n, int N1, int N2)
{
	mat N(2,2,fill::zeros);
	N(0,0)=1;

	mat M=kron(kron(kron(I_mat(N1),N), kron(I_mat(N2-N1-1),N) ), I_mat(n-N2-1));
	return M;
}

mat hopp_mat(int n, int n1, int n2)
{
	mat Cp(2,2,fill::zeros), Cm(2,2,fill::zeros);
	Cp(0,1)=1;	Cm(1,0)=1;	

	mat h1=kron( I_mat(n1), kron( kron(Cm,I_mat(n2-n1-1)), kron(Cp,I_mat(n-n2-1)) ) ); 
	mat h2=kron( I_mat(n1), kron( kron(Cp,I_mat(n2-n1-1)), kron(Cm,I_mat(n-n2-1)) ) );
	mat h=h1+h2;

	return h;
}

mat DC_mat(int L, double t1, double t2, double U, double V)
{
	int n=4*L;
	//int dim=1<<n;

	int n0=0, n1=L, n2=2*L, n3=3*L;

	mat H =U*( NiNj(n,n0,n1) + NiNj(n,n2,n3));
	   	H+=V*( NiNj(n,n0,n2) + NiNj(n,n0,n3) + NiNj(n,n1,n2) + NiNj(n,n1,n3) );
	

	for(int i=1;i<L;i++) 
	{
		for(int j=0;j<4;j++) H+=((j<2) ? t1 : t2)*hopp_mat(n,(i-1)+j*L,i+j*L);

		n0=i; n1=i+L; n2=i+2*L; n3=i+3*L;

	 	H+=U*( NiNj(n,n0,n1) + NiNj(n,n2,n3));
	   	H+=V*( NiNj(n,n0,n2) + NiNj(n,n0,n3) + NiNj(n,n1,n2) + NiNj(n,n1,n3) );
	}

	return(H);	
}




double DMRG(MPS H, int m, int nloops)
{
	int n=H.n, d=sqrt(H.d), m_H=H.m;
	H.canonize(1);

	MPS S(n, m, d); 
	S.canonize(0); 
	S.normalize();

	vector<cube> L, R;
	L.resize(n); R.resize(n);
	cube Hcube(d*d,m_H,1), Scube(d,m,1);

	for(int i=0; i<n; i++) {L[i]=cube(m,m_H,m); R[i]=cube(m,m_H,m);} 

	contract_0( R[n-2], S.sf, S.sf, H.sf, m, m_H, d, false); 
	for(int i=n-3; i>=0; i--) contract_step(R[i+1], R[i], S.C[i], S.C[i], H.C[i], m, m_H, d, false);
 	
	for(int lap=0; lap<nloops; lap++)
	{
        if(lap%1==0) cout<<"############################## "<<"[sweep = "<<lap<<"]"<<endl; 

		Hcube.reshape(d*d,m_H,1); 	Scube.reshape(d,m,1);
		Hcube.slice(0)=H.s0; 		Scube.slice(0)=S.s0;  

		Lanczos(R[0], R[0], Hcube, Scube, true, true); 
		S.sweep(true); 
		contract_0( L[1], S.s0, S.s0, H.s0, m, m_H, d, true);


		for(int site=1; site<n-1; site++)
		{
            cout<<"(sitio = "<<site<<")\t";

            cout<< sandwich(S, S, H) <<endl;

			Lanczos(L[site], R[site], H.C[site-1], S.C[site-1], false);
			S.sweep(true); 
			contract_step(L[site], L[site+1], S.C[site-1], S.C[site-1], H.C[site-1], m, m_H, d, true);
		}

		Hcube.reshape(m_H,d*d,1); 	Scube.reshape(m,d,1);
		Hcube.slice(0)=H.sf;		Scube.slice(0)=S.sf;
		
		Lanczos(L[n-1], L[n-1], Hcube, Scube, true, false);
		S.sweep(false);	
		contract_0( R[n-2], S.sf, S.sf, H.sf, m, m_H, d, false);

		for(int site=n-2; site>0; site--)
		{
            cout<<"(sitio = "<<site<<")\t";

            cout<< sandwich(S, S, H) <<endl;

			Lanczos(L[site], R[site], H.C[site-1], S.C[site-1], false);
			S.sweep(false);	
			contract_step(R[site], R[site-1], S.C[site-1], S.C[site-1], H.C[site-1], m, m_H, d, false);
		}

		
	}

	S.sweep(true);	
	return(sandwich(S, S, H));
}






int main(int argc, char const *argv[])
{
    srand(time(0));
    arma_rng::set_seed(rand());


    int L=3;

    double t1=1, t2=1, U=0, V=0;
    
    int mMPS=10, nsweeps=10;

    if(argc==2) mMPS= atof(argv[1]);




    MPS H=DC(L, t1, t2, U, V);
    //MPS H=TB(L);
    


    cout<<endl<<"Energy = \t"<<DMRG(H, mMPS, nsweeps)<<endl;

    cout<<endl<<"4 * TB_exact = \t"<<4*tb_exact(L)<<endl;

    return 0;
}


