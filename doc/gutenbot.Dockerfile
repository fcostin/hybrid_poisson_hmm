FROM thomasweise/docker-texlive-thin:1.0
RUN mkdir -p /work
ENV TEXINPUTS="/work/src:${TEXINPUTS}"
WORKDIR /work
