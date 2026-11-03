"""Exact source modules on synthetic embeddings; no dataset or training step."""
import sys
sys.dont_write_bytecode = True
import json
import gc
import argparse
import torch
from torch import nn
import reference_models as a

torch.set_num_threads(1)

def copy(p,u):
    p.load_state_dict(u.state_dict(),strict=True)

def tensor(p,u):
    assert p.shape==u.shape,(p.shape,u.shape)
    p.copy_(u)

def tokenizer(p,u,token_order=None,input_order=None):
    # The actual U tokenizer uses MLP_Block; P uses an equivalent Sequential.
    order=range(len(u.token_mlps)) if token_order is None else token_order
    for pm,j in zip(p.token_mlps,order,strict=True):
        um=u.token_mlps[j]
        copy(pm,um.mlp)
        if input_order is not None:
            tensor(pm[0].weight,um.mlp[0].weight[:,input_order])

def ff(p,u):
    # Map U Linear/activation/Linear into P Linear/activation/Dropout/Linear.
    copy(p.net[0],u[0]);copy(p.net[3],u[2])

def perff(p,u):
    for pn,un in [('first_weight','W1'),('first_bias','b1'),('second_weight','W2'),('second_bias','b2')]:
        tensor(getattr(p,pn),getattr(u,un))

def glu(p,u):
    for pn,un in [('w1','W1'),('w2','W2'),('w3','W3'),('b1','b1'),('b2','b2')]:
        if hasattr(p,pn) and hasattr(u,un):
            x,y=getattr(p,pn),getattr(u,un)
            if isinstance(x,nn.Module):copy(x,y)
            else:tensor(x,y)

def projections(p,u):
    for pn,un in [('query','q_proj'),('key','k_proj'),('value','v_proj'),('output','out_proj')]:
        copy(getattr(p,pn),getattr(u,un))

def semantic(name):
    return {'item':'item_id','candidate_action':'action','event':'action'}.get(name,name)

def align(b,h,u,c,family):
    d=c.backbone.dim;e=c.embedding_dim
    ef=list(c.feature_to_encoder(event=True))
    uf=[n for n,s in u.feature_map.features.items() if s.get('source') in ['item','action']]
    uc=[n for n,s in u.feature_map.features.items() if s.get('source') not in ['item','action']]+uf
    cf=list(c.feature_to_encoder(context=True))
    perm=[uc.index(semantic(n)) for n in cf]
    flatperm=[j*e+k for j in perm for k in range(e)]
    with torch.no_grad():
        if b.event_projection is not None:
            for i,n in enumerate(ef):
                j=uf.index(semantic(n))
                tensor(b.event_projection.fields[i].weight,u.item_token_proj.weight[:,j*e:(j+1)*e])
                b.event_projection.fields[i].bias.zero_()
            tensor(b.event_projection.fields[0].bias,u.item_token_proj.bias)
        else:
            event=b.event_aggregator.projection
            for i,n in enumerate(ef):
                j=uf.index(semantic(n))
                tensor(event.weight[:,i*e:(i+1)*e],u.item_token_proj.weight[:,j*e:(j+1)*e])
            tensor(event.bias,u.item_token_proj.bias)
        for pm,um in zip(h.layers,u.tower,strict=True): copy(pm,um.mlp)
        history=b.history_aggregator;us=u.unified_interaction_layers
        if family in ['est','hyformer','mixformer']:
            ut=u.ns_tokenizer if family=='hyformer' else u.unified_tokenizer_layer
            tokenizer(b.context_aggregator,ut,token_order=perm if family=='est' else None)
        if family=='est':
            headperm=[j*d+k for j in perm for k in range(d)]+list(range(len(cf)*d,(len(cf)+1)*d))
            for pm,um in zip(h.layers,u.tower,strict=True):tensor(pm[0].weight,um.mlp[0].weight[:,headperm])
            for i in range(3):
                cross=history.interaction.layers[i]
                seq=history.sequence.layers[i]
                ctx=history.nonsequence.layers[i]
                ulca=us.lca_layers[i];uff=us.ffn_layers[i]
                copy(cross.attention_norm,us.attention_norms[i])
                tensor(cross.query_weight,ulca.W_q_ns[perm]);tensor(cross.output,ulca.W_o[perm])
                copy(cross.key,ulca.W_k_s);copy(cross.value,ulca.W_v_s)
                assert seq.blocks[0]._norm[0] is ctx.blocks[0]._norm[0]
                copy(seq.blocks[0]._norm[0],us.ffn_norms[i])
                ff(seq.blocks[0].op,uff.ffn_s)
                for pn,un in [('first_weight','W1_ns'),('first_bias','b1_ns'),('second_weight','W2_ns'),('second_bias','b2_ns')]:
                    tensor(getattr(ctx.blocks[0].op,pn),getattr(uff,un)[perm])
        elif family=='hyformer':
            tokenizer(history.nonsequence.query_generator,u.query_generator,input_order=flatperm+list(range(len(cf)*e,len(cf)*e+d)))
            for i in range(3):
                seq=history.sequence.layers[i]
                cross=history.interaction.layers[i]
                boost=history.nonsequence.layers[i]
                useq,uq,ub=us.seq_layers[i],us.decode_layers[i],us.boost_layers[i]
                attn,ffn=seq.blocks
                copy(attn.norm,useq.norm);projections(attn.op,useq);ff(ffn.op,useq.ffn)
                copy(cross.norm,uq.norm);projections(cross,uq)
                copy(boost.blocks[0].norm,ub.mixer_norm)
                copy(boost.blocks[1].norm,ub.pffn_norm)
                perff(boost.blocks[1].op,ub.pffn)
        elif family=='mixformer':
            for i in range(3):
                mix,qff,out=history.nonsequence.layers[i]
                seq=history.sequence.layers[i]
                cross=history.interaction.layers[i]
                umix,ucross,uout=us.query_mixers[i],us.cross_attentions[i],us.output_fusions[i]
                copy(mix.blocks[0].norm,umix.norm1)
                copy(qff.blocks[0].norm,umix.norm2);glu(qff.blocks[0].op,umix.per_head_ffn)
                copy(seq.blocks[0].norm,ucross.seq_norm);glu(seq.blocks[0].op.net,ucross.seq_ffn)
                copy(cross.key,ucross.k_proj);copy(cross.value,ucross.v_proj)
                copy(out.blocks[0].norm,uout.norm);glu(out.blocks[0].op,uout.per_head_ffn)
        elif family=='hemix':
            t=history.interaction.pools[0].tokenizer;ut=u.unified_tokenizer_layer
            tokenizer(t,ut.token_mlps,input_order=flatperm)
            tensor(t.cls_tokens_earlier,ut.cls_tokens_glocal)
            tensor(t.cls_tokens_recent,ut.cls_tokens_real)
            for p_att,u_att in [(t.attention_earlier,ut.mha_glocal),(t.attention_recent,ut.mha_real)]:
                tensor(p_att.query_weight,u_att.W_q_ns)
                for pn,un in [('key','W_k_s'),('value','W_v_s'),('output','W_o')]:copy(getattr(p_att,pn),getattr(u_att,un))
            for i in range(3):
                mix,ffn=history.nonsequence.layers[0].blocks[2*i:2*i+2]
                copy(mix.norm,us.mixer_norms[i]);copy(ffn.norm,us.pffn_norms[i]);perff(ffn.op,us.pffn_layers[i])
                for pn,un in [('left_weight','W_l'),('left_bias','b_l'),('right_weight','W_r'),('right_bias','b_r')]:
                    tensor(getattr(mix.op,pn),getattr(us.mixer_layers[i],un))

def synthetic(c,u,length,regime):
    batch=2;e=c.embedding_dim
    lengths={'warm':[length,length], 'mixed':[3,length], 'cold':[0,0]}[regime]
    valid=torch.arange(length)[None,:]>=length-torch.tensor(lengths)[:,None]
    uf=[n for n,s in u.feature_map.features.items() if s.get('source') in ['item','action']]
    un=[n for n,s in u.feature_map.features.items() if s.get('source') not in ['item','action']]
    hist={n:(torch.randn(batch,length,e)*valid[:,:,None]).requires_grad_() for n in uf}
    ctx={n:torch.randn(batch,e).requires_grad_() for n in un+uf}
    # U's target action is padding; this test bypasses embedding-ID semantics.
    with torch.no_grad():ctx['action'].zero_()
    ui=({n:ctx[n] for n in un},{n:torch.cat([hist[n],ctx[n][:,None]],dim=1) for n in uf},valid)
    pe={n:hist[semantic(n)] for n in c.feature_to_encoder(event=True)}
    pc={n:ctx[semantic(n)] for n in c.feature_to_encoder(context=True)}
    # Positive recency positions; no event-store or dataset involved.
    pos=torch.where(valid,torch.arange(length,0,-1)[None,:],0).to(torch.int32)
    return ui,pe,pc,pos,{**{'history/'+n:v for n,v in hist.items()},**{'context/'+n:v for n,v in ctx.items()}}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dim',type=int,default=24)
    parser.add_argument('--out-dir',required=True)
    args=parser.parse_args()
    a.OUT = __import__('pathlib').Path(args.out_dir)
    a.OUT.mkdir(parents=True, exist_ok=True)
    results=[]
    for ds in ['taobao','merrec']:
        for fam in ['est','hyformer','mixformer','hemix']:
          variants=['reference_order_control'] if ds=='merrec' and fam in ['hyformer','mixformer'] else ['configured']
          for variant in variants:
            _,c=a.pconfig(ds,fam);c=c.model_copy(deep=True);c.backbone.dim=args.dim
            if variant=='reference_order_control':
                fm=a.fmap(ds)
                raworder=[n for n,s in fm.features.items() if s.get('source') not in ['item','action']]+[n for n,s in fm.features.items() if s.get('source') in ['item','action']]
                reverse={semantic(n):n for n in c.feature_to_encoder(context=True)}
                order=[reverse[n] for n in raworder]
                c.features={**{n:c.features[n] for n in order},**{n:v for n,v in c.features.items() if n not in order}}
            uc=a.uconfig(ds,fam)|{'token_dim':args.dim}
            torch.manual_seed(20261025)
            b,h=a.pmodel(c, ds)
            u=getattr(a.UMODULES[fam],a.NAMES[fam])(a.fmap(ds),**uc)
            b.eval();h.eval();u.eval();align(b,h,u,c,fam)
            for regime in ['warm','mixed','cold']:
                torch.manual_seed(1901)
                ui,pe,pc,pos,leaves=synthetic(c,u,24,regime)
                uy=u(ui)
                uz=torch.cat([uy[n+'_pred'] for n in u.feature_map.labels],dim=-1)
                emb={name:(lambda x:x) for name in set(pe)|set(pc)}
                pstate=b(emb,pe,pos,pos.float(),torch.ones(2)*25,pc)
                pz=torch.cat(h(pstate,None),dim=-1).sigmoid()
                scale=torch.arange(1,uz.size(1)+1)[None,:]
                ug=torch.autograd.grad((uz*scale).sum(),list(leaves.values()),allow_unused=True,retain_graph=True)
                pg=torch.autograd.grad((pz*scale).sum(),list(leaves.values()),allow_unused=True,retain_graph=True)
                labels=(torch.arange(uz.numel()).reshape_as(uz)%2).to(uz.dtype)
                uloss=torch.nn.functional.binary_cross_entropy(uz,labels,reduction='none').sum(dim=1).mean()
                ploss=torch.nn.functional.binary_cross_entropy(pz,labels,reduction='none').sum(dim=1).mean()
                up=torch.autograd.grad(uloss,[u.item_token_proj.weight,u.item_token_proj.bias],retain_graph=True)
                event_fields=list(c.feature_to_encoder(event=True))
                source_fields=[n for n,s in u.feature_map.features.items() if s.get('source') in ['item','action']]
                projection_deltas=[]
                if b.event_projection is not None:
                    fields=b.event_projection.fields
                    pp=torch.autograd.grad(ploss,[v for layer in fields for v in (layer.weight,layer.bias)],allow_unused=True)
                    for i,name in enumerate(event_fields):
                        j=source_fields.index(semantic(name));e=c.embedding_dim
                        projection_deltas.extend([float((pp[2*i]-up[0][:,j*e:(j+1)*e]).abs().max()),float((pp[2*i+1]-up[1]).abs().max())])
                else:
                    projection=b.event_aggregator.projection
                    pp=torch.autograd.grad(ploss,[projection.weight,projection.bias],allow_unused=True)
                    for i,name in enumerate(event_fields):
                        j=source_fields.index(semantic(name));e=c.embedding_dim
                        projection_deltas.append(float((pp[0][:,i*e:(i+1)*e]-up[0][:,j*e:(j+1)*e]).abs().max()))
                    projection_deltas.append(float((pp[1]-up[1]).abs().max()))
                grad_deltas=[]
                for (name,leaf),x,y in zip(leaves.items(),ug,pg,strict=True):
                    x=torch.zeros_like(leaf) if x is None else x
                    y=torch.zeros_like(leaf) if y is None else y
                    grad_deltas.append(dict(input=name,max_abs_delta=(x-y).abs().max().item(),
                        allclose=bool(torch.allclose(x,y,atol=1e-6,rtol=1e-4))))
                uorder=[n for n,s in u.feature_map.features.items() if s.get('source') not in ['item','action']]+[n for n,s in u.feature_map.features.items() if s.get('source') in ['item','action']]
                r=dict(dataset=ds,family=fam,variant=variant,regime=regime,token_dim=args.dim,embedding_dim=c.embedding_dim,
                    raw_context_order_equal=[semantic(n) for n in c.feature_to_encoder(context=True)]==uorder,
                    history_length=24,U_output_shape=list(uz.shape),P_output_shape=list(pz.shape),
                    max_abs_output_delta=(uz-pz).abs().max().item(),
                    finite_U=bool(torch.isfinite(uz).all()),finite_P=bool(torch.isfinite(pz).all()),
                    allclose=bool(torch.allclose(uz,pz,atol=1e-6,rtol=1e-5)),
                    input_gradients_allclose=all(g['allclose'] for g in grad_deltas),
                    input_gradient_max_abs_delta=max(g['max_abs_delta'] for g in grad_deltas),
                    input_gradient_deltas=grad_deltas,
                    task_loss_U=float(uloss.detach()),task_loss_P=float(ploss.detach()),task_loss_allclose=bool(torch.allclose(uloss,ploss,atol=1e-6,rtol=1e-5)),
                    projection_gradients_allclose=max(projection_deltas)<1e-6,projection_gradient_max_abs_delta=max(projection_deltas),
                    mapping='Field-token, head and dense-flat projection weights permuted by semantic field. Chunk tokenizer kept as implemented: no arbitrary cross-chunk weight remapping.')
                results.append(r);print(ds,fam,variant,regime,r['max_abs_output_delta'],r['input_gradient_max_abs_delta'],flush=True)
            del b,h,u
            gc.collect()
    report=dict(scope='Whole dense forward from supplied synthetic embeddings to probabilities. Actual source forwards; original U infrastructure bypassed. Original feature count/order, task count, depth, heads, FFN expansions and pooling windows. No vocabulary/ID/training equivalence claim. Event projection weights mapped with U single bias assigned to one P field; extra P biases zero.',
                token_dim=args.dim, torch_version=torch.__version__,
                padding_configuration='defined by the selected model configs',
                results=results)
    name='full_toy_equivalence.json' if args.dim==24 else 'full_dense_'+str(args.dim)+'.json'
    (a.OUT/name).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
