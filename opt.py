class Options:
    """
    학습 및 모델 하이퍼파라미터 설정 클래스
    argparse를 통해 명령행 인자로 설정값을 받거나 기본값 사용
    """
    
    def __init__(self):
        pass

    def init(self, parser):
        ''' ================================================전역 학습 설정================================================ '''
        parser.add_argument('--batch_size', type=int, default=128, #성능 안나오면 (split method를 바꿨기 때문 chazal split으로)  512로 바꿀예정
                            help='배치 크기')
        parser.add_argument('--nepoch', type=int, default=50,
                            help='전체 학습 에포크 수')
        parser.add_argument('--lr_initial', type=float, default=1e-4,
                            help='초기 학습률')
        parser.add_argument('--decay_epoch', type=int, default=20,
                            help='학습률 감소 시작 에포크')
        parser.add_argument('--device', type=str, default='cuda',
                            help='학습 디바이스 (cuda 또는 cpu)')

        ''' ================================================모델 기본 설정================================================ '''
        parser.add_argument('--classes', type=int, default=5,
                            help='분류 클래스 개수')
        parser.add_argument('--inputs', type=int, default=1,
                            help='입력 채널 수 (ECG lead 수)')
        parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                            help='저장된 모델 체크포인트 경로')

        ''' ================================================ResUNet 아키텍처 설정================================================ '''
        parser.add_argument('--out_ch', type=int, default=180,
                            help='출력 채널 수')
        parser.add_argument('--mid_ch', type=int, default=30,
                            help='Residual U-block 내부 채널 수')
        parser.add_argument('--inconv_size', type=int, default=5,
                            help='초기 컨볼루션 커널 크기')
        parser.add_argument('--r0_layer', type=int, default=3,
                            help='Residual U-block 레이어 수')

        ''' ================================================Proxy Loss 설정================================================ '''
        parser.add_argument('--proxy_weight', type=float, default=0,
                            help='Proxy loss 가중치')
        parser.add_argument('--cross_entropy_weight', type=float, default=1,
                            help='Cross entropy loss 가중치')

        ''' ================================================loss 별 비교 CE vs Proxy vs Focal Style Proxy vs Multi Proxy (proxy loss 사용시 결합 된 버전, 단독 사용은 proxy inference 사용    )================================================ '''
        parser.add_argument('--loss_type', type=str, default='CE',
                            help='손실 함수 타입')
        parser.add_argument('--proxy_type', type=str, default='FocalStyleProxyAnchorLoss',
                            choices=['ProxyAnchorLoss', 'FocalStyleProxyAnchorLoss', 'MultiProxyAnchorLoss'],
                            help='Proxy loss 타입')
        ''' ================================================결합여부 (단독 사용은 proxy inference 사용)================================================ '''
        parser.add_argument('--proxy_combined', type=bool, default=False,
                            help='Proxy loss 결합 여부, 이때 결합은 CE + proxy 형태임, false인 경우 proxy inference 사용  (단독 사용은 proxy inference 사용) ')
        parser.add_argument('--proxy_alpha', type=float, default=32.0,
                            help='Proxy loss alpha 파라미터')
        parser.add_argument('--proxy_delta', type=float, default=0.1,
                            help='Proxy loss delta 파라미터')
        parser.add_argument('--proxy_pos_gamma', type=float, default=2.0,
                            help='Proxy loss pos gamma 파라미터')
        parser.add_argument('--proxy_neg_gamma', type=float, default=2.0,
                            help='Proxy loss neg gamma 파라미터')
        
        ''' ================================================멀티 프록시인 경우 파라미터================================================ '''
        parser.add_argument('--multi_proxy_num_proxies_per_class', type=int, default=1,
                            help='Multi Proxy 각 클래스당 proxy 개수')
        parser.add_argument('--multi_proxy_topk_pos', type=int, default=None,
                            help='Multi Proxy hard mining을 위한 top-k')
        parser.add_argument('--multi_proxy_topk_neg', type=int, default=None,
                            help='Multi Proxy hard mining을 위한 top-k')
        parser.add_argument('--multi_proxy_softplus_threshold', type=float, default=20.0,
                            help='Multi Proxy softplus threshold')
    



        ''' ================================================데이터 경로 설정================================================ '''
        parser.add_argument('--path_train_data', type=str,
                            default='./data/processed/Exp_A1_1sec/train/train_data.npy',
                            help='학습 데이터 경로')
        parser.add_argument('--path_train_labels', type=str,
                            default='./data/processed/Exp_A1_1sec/train/train_labels.npy',
                            help='학습 레이블 경로')
        parser.add_argument('--path_test_data', type=str,
                            default='./data/processed/Exp_A1_1sec/test/test_data.npy',
                            help='테스트 데이터 경로')
        parser.add_argument('--path_test_labels', type=str,
                            default='./data/processed/Exp_A1_1sec/test/test_labels.npy',
                            help='테스트 레이블 경로')

        return parser