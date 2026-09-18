/* Opt-in real-model test: held session, batch/decode transitions, cancellation,
 * rebuild and snapshot replay. Compare stdout across revisions as an oracle. */
#include "ds4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s (%s)\n",__LINE__,#x,error); exit(1); } } while(0)
static char error[512];
static bool cancel_now(void *unused) { (void)unused; return true; }
static int finite_argmax(ds4_session *s, float *logits, int vocab) {
    CHECK(ds4_session_copy_logits(s, logits, vocab) == vocab);
    for (int i=0;i<vocab;i++) CHECK(isfinite(logits[i]));
    int id=ds4_session_argmax(s); CHECK(id>=0); return id;
}
int main(int argc,char **argv) {
    if(argc!=2) return 2;
    ds4_engine *e=NULL; ds4_session *s=NULL;
    ds4_engine_options o={.model_path=argv[1],.backend=DS4_BACKEND_METAL,
        .context_size=2048,.prefill_chunk=128,.ssd_streaming=true,.ssd_streaming_cache_experts=192};
    CHECK(ds4_engine_open(&e,&o)==0);
    CHECK(ds4_session_create(&s,e,2048)==0);
    int vocab=ds4_engine_vocab_size(e); float *logits=malloc((size_t)vocab*sizeof(float)); CHECK(logits);
    ds4_tokens prompt={0};
    ds4_encode_chat_prompt(e,NULL,"What is 17 times 23? Answer briefly.",DS4_THINK_NONE,&prompt);
    CHECK(ds4_session_sync(s,&prompt,error,sizeof(error))==0);
    for(int turn=0;turn<3;turn++) {
        printf("turn %d:",turn);
        for(int i=0;i<8;i++) {
            int id=finite_argmax(s,logits,vocab); printf(" %d",id);
            CHECK(ds4_session_eval(s,id,error,sizeof(error))==0);
            ds4_tokens_push(&prompt,id);
        }
        puts(""); fflush(stdout);
        if(turn<2) {
            ds4_chat_append_message(e,&prompt,"user",turn==0 ? "Now subtract 1 from that result." : "日本語で答えてください。");
            ds4_chat_append_think_prefix(e,&prompt,DS4_THINK_NONE);
            CHECK(ds4_session_sync(s,&prompt,error,sizeof(error))==0);
        }
    }
    ds4_session_snapshot snap={0};
    CHECK(ds4_session_save_snapshot(s,&snap,error,sizeof(error))==0);
    int first=finite_argmax(s,logits,vocab);
    CHECK(ds4_session_eval(s,first,error,sizeof(error))==0);
    int next=finite_argmax(s,logits,vocab);
    CHECK(ds4_session_load_snapshot(s,&snap,error,sizeof(error))==0);
    CHECK(finite_argmax(s,logits,vocab)==first);
    CHECK(ds4_session_eval(s,first,error,sizeof(error))==0);
    CHECK(finite_argmax(s,logits,vocab)==next);
    ds4_session_snapshot_free(&snap);
    ds4_tokens_free(&prompt);
    ds4_encode_chat_prompt(e,NULL,"Say hello in Japanese.",DS4_THINK_NONE,&prompt);
    ds4_session_set_cancel(s,cancel_now,NULL);
    CHECK(ds4_session_sync(s,&prompt,error,sizeof(error))==DS4_SESSION_SYNC_INTERRUPTED);
    ds4_session_set_cancel(s,NULL,NULL);
    CHECK(ds4_session_sync(s,&prompt,error,sizeof(error))==0);
    printf("rebuilt: %d\n",finite_argmax(s,logits,vocab));
    ds4_tokens_free(&prompt); free(logits); ds4_session_free(s); ds4_engine_close(e);
    puts("PASS: finite multi-turn streaming, snapshot replay and cancellation recovery");
    return 0;
}
