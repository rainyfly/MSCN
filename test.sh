CUDA_VISIBLE_DEVICES=5 python main.py --model baseline --save baseline --print_model --pre_train pretrain/baseline.pt --test_only --save_results --chop
CUDA_VISIBLE_DEVICES=5 python main.py --model ddbpn --save ddbpn --print_model --pre_train pretrain/ddbpn.pt --test_only --save_results --chop
CUDA_VISIBLE_DEVICES=5 python main.py --model edsr --save edsr --print_model --pre_train pretrain/edsr.pt --test_only --save_results --chop
CUDA_VISIBLE_DEVICES=5 python main.py --model memnet --save memnet --print_model --pre_train pretrain/memnet.pt --test_only --save_results --chop
CUDA_VISIBLE_DEVICES=5 python main.py --model rcan --save rcan --print_model --pre_train pretrain/rcan.pt --test_only --save_results --chop
CUDA_VISIBLE_DEVICES=5 python main.py --model srfeat --save srfeat --print_model --pre_train pretrain/srfeat.pt --test_only --save_results --chop
CUDA_VISIBLE_DEVICES=5 python main.py --model carn --save carn --print_model --pre_train pretrain/carn.pt --test_only --save_results --chop
CUDA_VISIBLE_DEVICES=5 python main.py --model dsrn --save dsrn --print_model --pre_train pretrain/dsrn.pt --test_only --save_results --chop
CUDA_VISIBLE_DEVICES=5 python main.py --model idn --save idn --print_model --pre_train pretrain/idn.pt --test_only --save_results --chop
CUDA_VISIBLE_DEVICES=5 python main.py --model msrn --save msrn --print_model --pre_train pretrain/msrn.pt --test_only --save_results --chop
CUDA_VISIBLE_DEVICES=5 python main.py --model srdensenet --save srdensenet --print_model --pre_train pretrain/srdensenet.pt --test_only --save_results --chop

