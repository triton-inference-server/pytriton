#!/usr/bin/groovy

@Library('test-shared-library@1.19') _

pipeline {
    agent none

    parameters {
        string(
                name: 'TRITON_SERVER_CONTAINER_VERSION',
                defaultValue: '22.12',
                description: 'Version of the Triton Inference Server Container to be used. CUDA version inside the Triton Server Container should match the CUDA version in Driverless AI. e.g. 22.12',
        )
        string(
                name: 'PYTHON_VERSIONS',
                defaultValue: '3.8,3.11',
                description: 'Comma seperated list of Python versions that the backend stubs needs to be build for. e.g. 3.8,3.9,3.10,3.11',
        )
        booleanParam(name: 'UPLOAD_TO_S3', defaultValue: true, description: 'Upload artifacts to S3')
    }

    options {
        ansiColor('xterm')
        timestamps()
        disableConcurrentBuilds(abortPrevious: true)
    }

    stages {
        stage('1. Build Python Backend Stubs for Triton') {
            agent {
                label "linux && docker && DC"
            }
            steps {
                timeout(time: 20, unit: 'MINUTES') {
                    sh "make BUILD_NUMBER=${env.BUILD_ID} build"
                    archiveArtifacts 'pytriton/tritonserver/python_backend_stubs/**/python_backend_stub'
                    stash includes: 'pytriton/tritonserver/python_backend_stubs/**/python_backend_stub', name: 'python_backend_stubs'
                }
            }
            post {
                always {
                    cleanWs()
                }
            }
        }

        stage('3. Push to S3') {
            when {
                anyOf {
                    expression { return params.UPLOAD_TO_S3 }
                }
                beforeAgent true
            }
            agent {
                label "linux && docker && DC"
            }
            steps {
                deleteDir()
                unstash 'python_backend_stubs'
                script {
                    def version = "${params.TRITON_SERVER_CONTAINER_VERSION}"
                    s3upDocker('harbor.h2o.ai', "library/awscli-x86_64") {
                        localArtifact = 'pytriton/tritonserver/python_backend_stubs'
                        remoteArtifactBucket = 's3://artifacts.h2o.ai/deps/dai'
                        artifactId = 'triton/python_backend_stub'
                        version = version
                        keepPrivate = false
                    }
                    def links = params.PYTHON_VERSIONS.split(',')
                            .collect { pyVersion -> "https://s3.amazonaws.com/artifacts.h2o.ai/deps/dai/triton/python_backend_stubs/${version}/${pyVersion}/python_backend_stub" }
                            .collect { s3Url -> "<a href=\"${s3Url}\">${s3Url}</a>" }
                            .collect { link -> "<li>${link}</li>" }
                            .join()
                    def summary = "<h3>Upoaded artifacts</h3> <ul>${links}</ul>"
                    manager.createSummary("package.svg").appendText(text: summary, escapeHtml: false)
                }
            }
            post {
                always {
                    cleanWs()
                }
            }
        }
    }
}
