export CUDA_VISIBLE_DEVICES='0'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
echo $PYTORCH_CUDA_ALLOC_CONF

source ~/anaconda3/etc/profile.d/conda.sh

conda activate pt
timer_start=`date "+%Y-%m-%d %H:%M:%S"`

nohup bgc_prophet pipeline --genomesDir /data4/yaoshuai/Aspergillus/speedtest_100_faa/ \
    --threads 50 --batch_size 1536\
    --outputPath /data4/yaoshuai/Aspergillus/speedtest_output_100/ --device cuda \
    --modelPath ./dist/annotator.pt \
    --saveIntermediate --name speedtest_100 --threshold 0.5 --max_gap 2 --min_count 2 \
    --classifierPath ./dist/classifier.pt \
    --classify_t 0.5 \
    >./nohup/Aspergillus_speedtest_100.out &

wait
timer_end=`date "+%Y-%m-%d %H:%M:%S"`
duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "开始： $timer_start" >> ./speedtest_BGC_Prophet_100.out
echo "结束： $timer_end" >> ./speedtest_BGC_Prophet_100.out
echo "耗时： $duration" >> ./speedtest_BGC_Prophet_100.out