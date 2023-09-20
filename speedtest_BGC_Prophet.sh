export CUDA_VISIBLE_DEVICES='1'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
echo $PYTORCH_CUDA_ALLOC_CONF

source ~/anaconda3/etc/profile.d/conda.sh

conda activate pt
timer_start=`date "+%Y-%m-%d %H:%M:%S"`

nohup python -u src/utils/pipline.py --genomesDir /data4/yaoshuai/Aspergillus/speedtest/ \
    --threads 50 --batch_size 1536\
    --outputPath /data4/yaoshuai/Aspergillus/speedtest_output/ --device cuda \
    --modelPath ./modelSave/transformerEncoder_TD_loss/bS_32_dE_200_lR_0.0005_mL_128_d_320_nH_5_nEL_2_tdP_0.1_mdP_0.1_TD/transformerEncoder_Model_TD_28.pt \
    --saveIntermediate --name speedtest --threshold 0.5 --max_gap 2 --min_count 2 \
    --classifierPath ./modelSave/transformerClassifier/transformerClassifier_128_5_2_0.5_0.1_0.01_150_0.1_1.0/transformerClassifier_50.pt \
    --classify_t 0.5 \
    >./nohup/Aspergillus_speedtest.out &

wait
timer_end=`date "+%Y-%m-%d %H:%M:%S"`
duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "开始： $timer_start" >> ./speedtest_BGC_Prophet.out
echo "结束： $timer_end" >> ./speedtest_BGC_Prophet.out
echo "耗时： $duration" >> ./speedtest_BGC_Prophet.out