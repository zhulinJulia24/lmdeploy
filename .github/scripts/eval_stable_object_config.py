from mmengine.config import read_base
from opencompass.models import OpenAISDK

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import \
        ARC_c_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import \
        bbh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.CHARM.charm_reason_cot_only_gen_f7b7d3 import \
        charm_reason_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmmlu.cmmlu_0shot_cot_gen_305931 import \
        cmmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.drop.drop_openai_simple_evals_gen_3857b0 import \
        drop_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_4c31db import \
        GaokaoBench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import \
        gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_a58960 import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import \
        hellaswag_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_159614 import \
        humaneval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFEval.IFEval_gen_3321a3 import \
        ifeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.LCBench.lcbench_gen_5ff288 import \
        LCBench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import \
        math_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MathBench.mathbench_2024_gen_50a320 import \
        mathbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_gen_a447ff import \
        sanitized_mbpp_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_openai_simple_evals_gen_b618ea import \
        mmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import \
        mmlu_pro_datasets  # noqa: F401, E501
    # from opencompass.configs.datasets.nq.nq_open_1shot_gen_2e45e5 import \
    #     nq_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_cot_gen_d95929 import \
        race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.scicode.scicode_gen_085b98 import \
        SciCode_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_cot_gen_1d56df import \
        BoolQ_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.teval.teval_en_gen_1ac254 import \
        teval_datasets as teval_en_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.teval.teval_zh_gen_1ac254 import \
        teval_datasets as teval_zh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import \
        TheoremQA_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_bc5f21 import \
        triviaqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.wikibench.wikibench_gen_0978ad import \
        wikibench_datasets  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.bbh import \
        bbh_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.cmmlu import \
        cmmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.GaokaoBench import \
        GaokaoBench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mathbench_v1_2024 import \
        mathbench_2024_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu import \
        mmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu_pro import \
        mmlu_pro_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.scicode import \
        scicode_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.teval import \
        teval_summary_groups  # noqa: F401, E501

summarizer = dict(
    dataset_abbrs=[
        ['race-middle', 'accuracy'],
        ['race-high', 'accuracy'],
        ['ARC-c', 'accuracy'],
        ['BoolQ', 'accuracy'],
        ['mmlu_pro', 'naive_average'],
        ['drop', 'accuracy'],
        ['bbh', 'naive_average'],
        ['GPQA_diamond', 'accuracy'],
        ['math', 'accuracy'],
        ['wikibench-wiki-single_choice_cncircular', 'perf_4'],
        ['openai_humaneval', 'humaneval_pass@1'],
        ['sanitized_mbpp', 'score'],
        ['cmmlu', 'naive_average'],
        ['mmlu', 'naive_average'],
        ['teval', 'naive_average'],
        ['SciCode', 'accuracy'],
        ['SciCode', 'sub_accuracy'],
        ['IFEval', 'Prompt-level-strict-accuracy'],
        ['gsm8k', 'accuracy'],
        ['GaokaoBench', 'weighted_average'],
        ['triviaqa_wiki_1shot', 'score'],
        ['nq_open_1shot', 'score'],
        ['hellaswag', 'accuracy'],
        ['TheoremQA', 'score'],
        '###### MathBench-A: Application Part ######',
        'college',
        'high',
        'middle',
        'primary',
        'arithmetic',
        'mathbench-a (average)',
        '###### MathBench-T: Theory Part ######',
        'college_knowledge',
        'high_knowledge',
        'middle_knowledge',
        'primary_knowledge',
        'mathbench-t (average)',
        '###### Overall: Average between MathBench-A and MathBench-T ######',
        'Overall',
        '',
        ''
        'mmlu',
        'mmlu-stem',
        'mmlu-social-science',
        'mmlu-humanities',
        'mmlu-other',
        '',
        'cmmlu',
        'cmmlu-stem',
        'cmmlu-social-science',
        'cmmlu-humanities',
        'cmmlu-other',
        'cmmlu-china-specific',
        '',
        'mmlu_pro',
        'mmlu_pro_biology',
        'mmlu_pro_business',
        'mmlu_pro_chemistry',
        'mmlu_pro_computer_science',
        'mmlu_pro_economics',
        'mmlu_pro_engineering',
        'mmlu_pro_health',
        'mmlu_pro_history',
        'mmlu_pro_law',
        'mmlu_pro_math',
        'mmlu_pro_philosophy',
        'mmlu_pro_physics',
        'mmlu_pro_psychology',
        'mmlu_pro_other',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')
                and 'scicode' not in k.lower() and 'teval' not in k), [])
datasets += teval_en_datasets
datasets += teval_zh_datasets
datasets += SciCode_datasets

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        abbr='lmdeploy-api-test',
        type=OpenAISDK,
        key='EMPTY',
        openai_api_base='http://localhost:23344/v1',
        path='/nvme/qa_test_models/internlm/internlm2_5-20b-chat',
        tokenizer_path='/nvme/qa_test_models/internlm/internlm2_5-20b-chat',
        rpm_verbose=True,
        meta_template=api_meta_template,
        query_per_second=100,
        max_out_len=1024,
        max_seq_len=4096,
        temperature=0.01,
        batch_size=128,
        retry=3,
    )
]
