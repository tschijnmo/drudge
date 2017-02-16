#!/usr/bin/env bash

#
# Run the given drudge script in a SLURM job.
#
# The actual script and its command line arguments should be given to this
# auxiliary script.
#
# Before calling this script, the environmental variables SPARK_HOME,
# JAVA_HOME, and PYTHONPATH need to be set correctly.  And the program
# `python3` in PATH need to point to the Python interpreter intended to be
# used.
#
# SPARK_LOG_LEVEL can be used to tune the logging level for Spark, by default,
# only errors are logged due to performance reasons.
#
# MEM_AMOUNT can be used to tune the memory amount for both the driver and
# worker during spark-submit.  Normally this does not need to be set.  A
# symbolic value of ALL can be used to use up all available memory.
#

if [ -z "${SPARK_HOME}" ]; then
    echo "SPARK_HOME is not set!"
    exit 1
fi

if [ "$#" -lt 1 ]; then
    echo "No script is given!"
    exit 1
fi


#
# Create the directories needed by Spark for the job.
#

spark_job_home="Spark-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir $spark_job_home

spark_conf_dir=${spark_job_home}/conf
mkdir ${spark_conf_dir}

spark_worker_dir=${spark_job_home}/work
mkdir ${spark_worker_dir}

spark_log_dir=${spark_job_home}/log
mkdir ${spark_log_dir}

spark_pid_dir=${spark_job_home}/pid
mkdir ${spark_pid_dir}


#
# Set the Spark configuration
#

spark_master_host=$(hostname)
spark_master_port=7077

export SPARK_CONF_DIR=${spark_conf_dir}

cat > ${spark_conf_dir}/spark-env.sh << EOF
export SPARK_LOG_DIR=${spark_log_dir}
export SPARK_PID_DIR=${spark_pid_dir}
export SPARK_WORKER_DIR=${spark_worker_dir}

export SPARK_MASTER_HOST=${spark_master_host}
export SPARK_MASTER_PORT=${spark_master_port}

export PYSPARK_PYTHON=$(which python3)
export PYSPARK_DRIVER_PYTHON=$(which python3)
EOF

SPARK_LOG_LEVEL=${SPARK_LOG_LEVEL:-ERROR}

cat > ${spark_conf_dir}/log4j.properties << EOF
log4j.rootCategory=${SPARK_LOG_LEVEL}, console
log4j.appender.console=org.apache.log4j.ConsoleAppender
log4j.appender.console.target=System.err
log4j.appender.console.layout=org.apache.log4j.PatternLayout
log4j.appender.console.layout.ConversionPattern=%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n

# Set the default spark-shell log level to WARN. When running the spark-shell, the
# log level for this class is used to overwrite the root logger's log level, so that
# the user can have different defaults for the shell and regular Spark apps.
log4j.logger.org.apache.spark.repl.Main=WARN

# Settings to quiet third party logs that are too verbose
log4j.logger.org.spark_project.jetty=WARN
log4j.logger.org.spark_project.jetty.util.component.AbstractLifeCycle=ERROR
log4j.logger.org.apache.spark.repl.SparkIMain$exprTyper=INFO
log4j.logger.org.apache.spark.repl.SparkILoop$SparkILoopInterpreter=INFO
log4j.logger.org.apache.parquet=ERROR
log4j.logger.parquet=ERROR

# SPARK-9183: Settings to avoid annoying messages when looking up nonexistent UDFs in SparkSQL with Hive support
log4j.logger.org.apache.hadoop.hive.metastore.RetryingHMSHandler=FATAL
log4j.logger.org.apache.hadoop.hive.ql.exec.FunctionRegistry=ERROR
EOF


#
# Start the Spark cluster
#

export_env="PATH,LD_LIBRARY_PATH,\
JAVA_HOME,SPARK_CONF_DIR,SPARK_NO_DAEMONIZE,\
PYTHONPATH,PYTHONHASHSEED"

if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then

    export PYTHONHASHSEED=323
    spark_master_link="spark://${spark_master_host}:${spark_master_port}"

    ${SPARK_HOME}/sbin/start-master.sh

    export SPARK_NO_DAEMONIZE=1

    srun --export="$export_env" \
    ${SPARK_HOME}/sbin/start-slave.sh ${spark_master_link} &

    unset SPARK_NO_DAEMONIZE

    sleep 30

else
    spark_master_link="local[*]"
fi


#
# Try to have a sensible setting of memory
#

MEM_AMOUNT=${MEM_AMOUNT:-DEFAULT}

if [ "$MEM_AMOUNT" = "ALL" ]; then
    if [ -z "$SLURM_MEM_PER_NODE" ]; then
        # For lower versions of SLURM where this is not set.
        MEM_AMOUNT=$(free -g | grep ^Mem: | awk '{print $2}')g
    else
        MEM_AMOUNT="$SLURM_MEM_PER_NODE"
    fi
fi

if [ "$MEM_AMOUNT" != "DEFAULT" ]; then
    mem_args="--executor-memory ${MEM_AMOUNT} --driver-memory ${MEM_AMOUNT}"
else
    mem_args=""
fi


#
# Run the given script.
#

echo "


********************************************************************************
Running Script $1 at ${spark_master_link}
$(date)
********************************************************************************


"

${SPARK_HOME}/bin/spark-submit --master "${spark_master_link}" \
${mem_args} \
"$@"

echo "


********************************************************************************
Script $1 finished
$(date)
********************************************************************************


"

