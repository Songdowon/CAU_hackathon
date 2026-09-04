# Same torch/cuda pin as the grading server (requirements_server.txt) --
# students should develop in the exact environment their submission gets
# scored in, so there are no last-minute "works on my machine" surprises.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# nginx + supervisor: same 3-process layout (nginx / FastAPI / code-server)
# as the original hackathon_api-dev-student image, ported to this task.
RUN apt-get update && apt-get install -y curl wget vim git nginx supervisor util-linux && \
    rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Build this Dockerfile from the repository root.  Participant-editable files
# and trusted control-plane files deliberately live in different image trees.
WORKDIR /opt/workspace-seed

COPY student_docker/requirements.txt ./
RUN pip install --no-cache-dir -r /opt/workspace-seed/requirements.txt

ARG PARTICIPANT_UID=10001
ARG PARTICIPANT_GID=10001
RUN groupadd --gid "${PARTICIPANT_GID}" participant && \
    useradd --uid "${PARTICIPANT_UID}" --gid participant -ms /bin/bash participant

# code + config: student-editable
COPY student_docker/imagenet_vit.py student_docker/train_ft.py student_docker/unlearn.py student_docker/validate_submission.py student_docker/score_model.py student_docker/walkthrough.ipynb student_docker/README.md ./
COPY student_docker/utils/ ./utils/
COPY student_docker/configs/ ./configs/
COPY student_docker/baselines/ ./baselines/
COPY student_docker/es/ ./es/

# The API, scoring implementation, and browser UI are trusted runtime components. They
# never enter /workspace, which is entirely participant-controlled.
WORKDIR /opt/hackathon
COPY student_docker/app/ ./app/
COPY student_docker/public/ ./public/
COPY grading_docker/score_unlearning.py grading_docker/convert_checkpoint.py grading_docker/imagenet_vit.py ./scorer/
COPY student_docker/nginx.conf /etc/nginx/nginx.conf
COPY student_docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN rm -f /etc/nginx/sites-enabled/default
RUN chown -R root:root /opt/hackathon && chmod -R go-w /opt/hackathon

# The v2 released + validation bundle is deliberately not baked into the
# image. One init service downloads it once; one copy job per team then puts an
# independent writable copy below that team's /workspace/datasets.
COPY student_docker/install_released_data.sh /usr/local/sbin/install-released-data
COPY student_docker/copy_released_data.sh /usr/local/sbin/copy-released-data
RUN chmod 755 /usr/local/sbin/install-released-data /usr/local/sbin/copy-released-data

RUN mkdir -p /var/log/supervisor /var/log/nginx /run

USER participant
RUN code-server --install-extension ms-python.python
RUN code-server --install-extension ms-toolsai.jupyter
USER root

# 80: nginx (main entrypoint) / 8000: FastAPI (direct, optional) /
# 8080: code-server (direct, optional)
EXPOSE 80 8000 8080

COPY student_docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Keep volatile provenance at the end so a new timestamp does not invalidate
# the expensive CUDA, apt, Python, and code-server build layers above.
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="unlearning-student" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

WORKDIR /opt/hackathon

# Runtime credentials are staged from root-protected *_FILE sources before
# supervisor starts; environment fallbacks exist only for explicit dev use.
ENTRYPOINT ["/entrypoint.sh"]
