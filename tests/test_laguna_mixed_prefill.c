#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/mman.h>
#include <unistd.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL line %d\n",__LINE__);exit(1);}}while(0)
typedef struct {uint16_t d,dmin;uint8_t scales[12],qs[128];} block;
static uint32_t seed=1;
static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
/* Regression: mixed Q4 gate/up and Q6 down must retain grouped arithmetic
 * when SSD streaming falls back to mapped expert weights. No model required. */
int main(void){
 enum{D=256,H=256,E=32,N=8,T=551};
 uint64_t row=sizeof(block),expert=H*row,tensor=E*expert,downrow=210,downexpert=H*downrow,bytes=2*tensor+E*downexpert;
 FILE *file=tmpfile();CHECK(file);CHECK(!ftruncate(fileno(file),bytes));
 block *model=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fileno(file),0);CHECK(model!=MAP_FAILED);
 for(size_t i=0;i<2*tensor/sizeof(block);i++){model[i].d=0x1800;model[i].dmin=0x1400;for(int j=0;j<12;j++)model[i].scales[j]=rnd();for(int j=0;j<128;j++)model[i].qs[j]=rnd();}
 typedef struct {uint8_t ql[128],qh[64];int8_t scales[16];uint16_t d;} q6;
 q6 *down=(q6*)((char*)model+2*tensor);for(size_t i=0;i<E*downexpert/sizeof(q6);i++){for(int j=0;j<128;j++)down[i].ql[j]=rnd();for(int j=0;j<64;j++)down[i].qh[j]=rnd();for(int j=0;j<16;j++)down[i].scales[j]=(rnd()%15)-7;down[i].d=0x1800;}
 CHECK(!msync(model,bytes,MS_SYNC));CHECK(ds4_gpu_init());ds4_gpu_set_quality(false);
 float *x=malloc(T*D*4),*w=malloc(T*N*4),*a=malloc(T*D*4),*b=malloc(T*D*4);int32_t *ids=malloc(T*N*4);
 for(int i=0;i<T*D;i++)x[i]=((int)(rnd()%101)-50)/256.f;
 for(int t=0;t<T;t++)for(int k=0;k<N;k++){ids[t*N+k]=(t*13+k*17)%E;w[t*N+k]=(k+1)/36.f;}
 ds4_gpu_tensor *xt=ds4_gpu_tensor_alloc(T*D*4),*wt=ds4_gpu_tensor_alloc(T*N*4),*it=ds4_gpu_tensor_alloc(T*N*4),*mid=ds4_gpu_tensor_alloc(T*N*H*4),*out=ds4_gpu_tensor_alloc(T*D*4);
 CHECK(xt&&wt&&it&&mid&&out);CHECK(ds4_gpu_tensor_write(xt,0,x,T*D*4));CHECK(ds4_gpu_tensor_write(wt,0,w,T*N*4));CHECK(ds4_gpu_tensor_write(it,0,ids,T*N*4));
 int failed=0; const int counts[]={95,96,97,257,551};
 for(unsigned c=0;c<sizeof(counts)/sizeof(counts[0]);c++){int n=counts[c];
 for(int stream=0;stream<2;stream++){
 ds4_gpu_set_ssd_streaming(stream);ds4_gpu_set_streaming_expert_cache_budget(16);ds4_gpu_set_streaming_expert_cache_expert_bytes(2*expert+downexpert);CHECK(ds4_gpu_set_model_map(model,bytes));CHECK(ds4_gpu_set_model_fd(fileno(file)));
 CHECK(ds4_gpu_begin_commands());CHECK(ds4_gpu_glm_routed_moe_batch_tensor(out,mid,model,bytes,0,tensor,2*tensor,12,12,14,expert,row,expert,row,downexpert,downrow,D,H,D,it,wt,E,N,0.f,0,xt,n,N*H,true));CHECK(ds4_gpu_end_commands());CHECK(ds4_gpu_tensor_read(out,0,stream?b:a,n*D*4));
 }
 float max=0;int bad=0;for(int i=0;i<n*D;i++){CHECK(isfinite(a[i])&&isfinite(b[i]));float d=fabsf(a[i]-b[i]);if(d>max)max=d;if(d!=0)bad++;}printf("tokens=%d max_diff=%g differing=%d\n",n,max,bad);failed|=bad!=0;}
 ds4_gpu_tensor_free(xt);ds4_gpu_tensor_free(wt);ds4_gpu_tensor_free(it);ds4_gpu_tensor_free(mid);ds4_gpu_tensor_free(out);ds4_gpu_cleanup();free(x);free(w);free(ids);free(a);free(b);munmap(model,bytes);fclose(file);return failed;
}
