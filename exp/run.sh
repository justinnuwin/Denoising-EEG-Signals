export TASK=baseline
for file in $(ls data)
do
    export MAT=$file
    python3 ${TASK}.py
done