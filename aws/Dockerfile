FROM public.ecr.aws/lambda/python:3.10

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# Copy function code
COPY main.py ${LAMBDA_TASK_ROOT}

# Copy package folder 
COPY package ${LAMBDA_TASK_ROOT}/package

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "main.run_model" ]
