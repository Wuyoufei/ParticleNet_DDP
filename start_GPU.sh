suffix=$1
rrun=$2
cwd=`pwd`
echo 'Please ensure the requirements are what you need: ';echo
sed -n '6,9p' run_GPU.sh;echo

if [ -z $suffix ]; then
    read -p 'please type a suffix for this training: ' suffix 
    echo "suffix is '$suffix'"
else
    echo "suffix is '$suffix'"
fi

while [ -e "./output/figs/pfn_conf_${suffix}.pdf" ]
do
    read -p "suffix '${suffix}' has been used, override it?(y/n): " option
    case ${option} in
        "y")
            echo "override suffix '${suffix}' confirmed"
            break
            ;;
        "n")
            read -p "reset your suffix: " suffix
            ;;
        *)
            echo 'please type y or n'
            ;;
    esac
done
        



sed -i "10c #SBATCH --output=${cwd}/output/my_log/PN_${suffix}.log" run_GPU.sh
sed -i "45c python -u script_GPUonly/my_train_DDP.py ${suffix}" run_GPU.sh

if [ "$rrun" == l ]; then
    echo 'running code locally(for test usually)'
    python -u script_GPUonly/my_train_DDP.py ${suffix}
    exit 0
elif [ "$rrun" == s ]; then
    sbatch run_GPU.sh
    exit 0
fi



while [ "$rrun" != l -a "$rrun" != s ]
do
    read -p 'run local or slurm (l/s): ' rrun
    if [ "$rrun" == l ]; then
        echo 'running code locally(for test usually)'
        python -u script_GPUonly/my_train_DDP.py ${suffix}
        exit 0
    elif [ "$rrun" == s ]; then
        sbatch run_GPU.sh
        exit 0
    fi
done












