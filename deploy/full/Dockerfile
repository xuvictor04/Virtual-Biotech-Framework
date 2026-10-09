# syntax=docker/dockerfile:1
# Harness image: the Python + R environment of the harness, the agents' Bash and the UNMODIFIED upstream MCP
# servers (pinned: deploy/full/conda-linux-64.lock + deploy/full/requirements.lock), the harness (editable, so
# it finds configs/ next to src/) and the upstream submodule at the commit the repository pins.
#
# Build from a clean clone with the submodule checked out (git clone --recursive, or git submodule update --init):
#   deploy/full/deploy.sh build                      # docker build -f deploy/full/Dockerfile -t vbt-harness:<rev> .
#   deploy/full/deploy.sh lock                       # regenerate the two lock files (target `lock`)
#
# Like the upstream Apptainer definition (third_party/TheVirtualBiotech/vbt.def): Ubuntu 22.04, micromamba in
# /usr/local/bin and one conda env under /opt/conda/envs/vbt. micromamba and a CA bundle are copied from the
# official micromamba image, so the build needs no apt and the env is created in place (its files are hard links
# to the package cache, which is removed in the same step). deploy/full/vbt-harness.def builds the same
# environment for Apptainer (HPC).
#
# Optional build inputs (all generic; none is needed on a host with direct internet access):
#   --build-arg HTTPS_PROXY=...           proxy for the package downloads
#   --secret id=extra_ca,src=FILE         PEM of a TLS-intercepting proxy's CA, trusted during the build only
#   --build-arg LOCKED=0                  resolve environment.yml + requirements.in instead of the locks
#                                         (LOCKED=conda: the conda lock, pip resolved from requirements.in)
#   --build-arg BASE_IMAGE=... / MICROMAMBA_IMAGE=...   another registry or a digest-pinned base
#
# Runtime layout (deploy/full/compose.yaml mounts these): /srv/vbt/{data,runs,projects,state,models}. The entrypoint
# (deploy/full/vbt-host) loads the host configuration `vbt setup` wrote to /srv/vbt/state and runs `vbt`.

ARG BASE_IMAGE=docker.io/library/ubuntu:22.04
ARG MICROMAMBA_IMAGE=docker.io/mambaorg/micromamba:2.3.2

# hadolint ignore=DL3006
FROM ${MICROMAMBA_IMAGE} AS micromamba

# ------------------------------------------------------------------------------------------------ the conda env
# hadolint ignore=DL3006
FROM ${BASE_IMAGE} AS env
ARG LOCKED=1
ARG HTTPS_PROXY
ARG https_proxy
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
COPY --from=micromamba /bin/micromamba /usr/local/bin/micromamba
COPY --from=micromamba /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY deploy/full/install-env.sh deploy/full/environment.yml deploy/full/requirements.in \
     deploy/full/conda-linux-64.lock deploy/full/requirements.lock /tmp/vbt-env/
RUN --mount=type=secret,id=extra_ca,required=false \
    set -eu; \
    if [ -s /run/secrets/extra_ca ]; then \
      cat /etc/ssl/certs/ca-certificates.crt /run/secrets/extra_ca > /tmp/ca-bundle.crt; \
      export SSL_CERT_FILE=/tmp/ca-bundle.crt REQUESTS_CA_BUNDLE=/tmp/ca-bundle.crt PIP_CERT=/tmp/ca-bundle.crt \
             MAMBA_SSL_VERIFY=/tmp/ca-bundle.crt CURL_CA_BUNDLE=/tmp/ca-bundle.crt; \
    fi; \
    case "$LOCKED" in 1) flag="" ;; conda) flag="--pip-unlocked" ;; *) flag="--unlocked" ;; esac; \
    bash /tmp/vbt-env/install-env.sh env --prefix /opt/conda/envs/vbt --spec-dir /tmp/vbt-env $flag; \
    rm -f /tmp/ca-bundle.crt

# ------------------------------------------------------------------- lock: `docker build --target lock --output DIR`
FROM env AS lockgen
RUN bash /tmp/vbt-env/install-env.sh export --prefix /opt/conda/envs/vbt --out /out
FROM scratch AS lock
COPY --from=lockgen /out/ /

# ------------------------------------------------------------------------------------------- runtime: the harness
FROM env AS harness
ARG VCS_REF=unknown
ARG UPSTREAM_COMMIT=unknown
ARG VERSION=0.1.0
LABEL org.opencontainers.image.title="vbt-harness" \
      org.opencontainers.image.description="Virtual Biotech harness: agents, data layer, unmodified upstream MCP servers" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="MIT" \
      io.vbt.upstream.commit="${UPSTREAM_COMMIT}"
ENV PATH=/opt/conda/envs/vbt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CONDA_PREFIX=/opt/conda/envs/vbt \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTMCP_SHOW_CLI_BANNER=false \
    VBT_HOME=/srv/vbt \
    VBT_HARNESS_COMMIT=${VCS_REF} \
    VBT_UPSTREAM_COMMIT=${UPSTREAM_COMMIT} \
    HOME=/home/vbt
RUN groupadd --gid 10001 vbt && useradd --uid 10001 --gid vbt --create-home --home-dir /home/vbt --shell /bin/bash vbt \
    && mkdir -p /srv/vbt/data /srv/vbt/runs /srv/vbt/projects /srv/vbt/state /srv/vbt/models \
    && chown -R vbt:vbt /srv/vbt \
    && rm -rf /tmp/vbt-env
COPY --chown=vbt:vbt . /opt/vbt
# The import check runs as root: its caches (numba, matplotlib) must not land in vbt's home, where root-owned
# directories would make numba's cache "no locator available" and break `import scanpy` at run time.
# The clone-relative defaults (paths.runs_dir: runs, data/...) land on the volumes even before `vbt setup` has run.
RUN HOME=/root bash /opt/vbt/deploy/full/install-harness.sh --prefix /opt/conda/envs/vbt --src /opt/vbt \
    && rm -rf /root/.cache /root/.config \
    && chmod 0755 /opt/vbt/deploy/full/vbt-host \
    && ln -s /srv/vbt/runs /opt/vbt/runs && ln -s /srv/vbt/data /opt/vbt/data
USER vbt
WORKDIR /opt/vbt
# No VOLUME: anonymous volumes would shadow a bind mount of the whole /srv/vbt (docker run -v $VBT_HOME:/srv/vbt).
# deploy/full/compose.yaml mounts data/, runs/, projects/ and state/ explicitly.
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/auth', timeout=5)"]
ENTRYPOINT ["tini", "-g", "--", "/opt/vbt/deploy/full/vbt-host"]
CMD ["web", "--host", "0.0.0.0", "--port", "7860"]
