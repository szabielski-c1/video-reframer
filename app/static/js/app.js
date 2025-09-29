// Video Reframer Web Interface JavaScript

class VideoReframer {
    constructor() {
        this.currentJobId = null;
        this.websocket = null;
        this.uploadedVideoUrl = null;
        this.healthCheckInterval = null;

        this.initializeEventListeners();
        this.setupDragAndDrop();
        this.startHealthMonitoring();
        this.loadRecentJobs();
    }

    initializeEventListeners() {
        // Form submission
        document.getElementById('upload-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleUpload();
        });

        // Cancel button
        document.getElementById('cancel-btn').addEventListener('click', () => {
            this.cancelJob();
        });

        // File input change
        document.getElementById('video-file').addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Range inputs - update display values
        document.getElementById('smoothing').addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = `Smoothing: ${e.target.value}`;
        });

        document.getElementById('padding').addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = `Padding: ${e.target.value}x`;
        });
    }

    setupDragAndDrop() {
        const uploadSection = document.getElementById('upload-section');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadSection.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadSection.addEventListener(eventName, () => {
                uploadSection.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadSection.addEventListener(eventName, () => {
                uploadSection.classList.remove('dragover');
            }, false);
        });

        uploadSection.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        }, false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('video/')) {
            this.showError('Please select a valid video file.');
            return;
        }

        // Validate file size (500MB limit)
        const maxSize = 500 * 1024 * 1024; // 500MB
        if (file.size > maxSize) {
            this.showError('File size exceeds 500MB limit. Please select a smaller video.');
            return;
        }

        // Store the file for later use
        this.selectedFile = file;

        // Preview the video
        this.previewVideo(file);
    }

    previewVideo(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.uploadedVideoUrl = e.target.result;

            // Create preview element if it doesn't exist
            let preview = document.getElementById('video-preview');
            if (!preview) {
                preview = document.createElement('video');
                preview.id = 'video-preview';
                preview.controls = true;
                preview.style.maxWidth = '100%';
                preview.style.maxHeight = '200px';
                preview.style.marginTop = '15px';
                preview.style.borderRadius = '10px';

                document.querySelector('#upload-form').appendChild(preview);
            }

            preview.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    async handleUpload() {
        const fileInput = document.getElementById('video-file');
        const file = this.selectedFile || fileInput.files[0];

        if (!file) {
            this.showError('Please select a video file to upload.');
            return;
        }

        try {
            this.showProgress();
            this.updateProgress(0, 'Uploading video...');

            // Create FormData for file upload
            const formData = new FormData();
            formData.append('video', file);

            // Add processing settings
            const settings = this.getProcessingSettings();
            formData.append('settings', JSON.stringify(settings));

            // Upload file and start processing
            const uploadResponse = await fetch('/api/v1/upload-and-reframe', {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                const errorData = await uploadResponse.json();
                throw new Error(errorData.detail || 'Upload failed');
            }

            const result = await uploadResponse.json();
            this.currentJobId = result.job_id;

            // Start WebSocket connection for real-time updates
            this.connectWebSocket(this.currentJobId);

        } catch (error) {
            console.error('Upload error:', error);
            this.showError(`Upload failed: ${error.message}`);
        }
    }

    getProcessingSettings() {
        return {
            mode: 'auto',
            style: document.getElementById('video-style').value,
            smoothing: parseFloat(document.getElementById('smoothing').value),
            padding: parseFloat(document.getElementById('padding').value),
            quality: document.getElementById('quality').value,
            preserve_text: document.getElementById('preserve-text').checked,
            enable_cuts: true,
            audio_analysis: true
        };
    }

    getGeminiPrompts() {
        const customFocus = document.getElementById('custom-focus').value;
        return {
            custom_focus: customFocus || null,
            priority_subjects: null,
            exclude_areas: null
        };
    }

    connectWebSocket(jobId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/${jobId}`;

        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
            console.log('WebSocket connected');
        };

        this.websocket.onmessage = (event) => {
            const update = JSON.parse(event.data);
            this.handleProgressUpdate(update);
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showError('Lost connection to server. Refreshing...');
            setTimeout(() => location.reload(), 3000);
        };

        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
        };
    }

    handleProgressUpdate(update) {
        console.log('Progress update received:', update);

        if (update.error) {
            this.showError(update.error);
            return;
        }

        this.updateProgress(update.progress, update.message);

        if (update.status === 'completed') {
            console.log('Processing completed, output_url:', update.output_url);
            console.log('Analytics received:', update.analytics);
            if (!update.output_url) {
                console.error('No output URL received in completion message:', update);
                this.showError('Processing completed but no video URL was provided. Please check the server logs.');
                return;
            }
            this.showResults(update.output_url, update.preview_url, update.analytics);
        } else if (update.status === 'failed') {
            this.showError(update.error || 'Processing failed');
        }
    }

    updateProgress(percentage, message) {
        document.getElementById('progress-percentage').textContent = `${Math.round(percentage)}%`;
        document.getElementById('progress-message').textContent = message;
        document.getElementById('progress-bar').style.width = `${percentage}%`;
        document.getElementById('job-id').textContent = this.currentJobId || '';
    }

    showProgress() {
        this.hideAllSections();
        document.getElementById('progress-section').classList.remove('d-none');
        document.getElementById('process-btn').disabled = true;
    }

    showResults(outputUrl, previewUrl, analytics) {
        if (!outputUrl) {
            console.error('showResults called without outputUrl');
            this.showError('No output video URL available');
            return;
        }

        this.hideAllSections();
        document.getElementById('results-section').classList.remove('d-none');

        // Set up videos
        if (this.uploadedVideoUrl) {
            document.getElementById('original-video').src = this.uploadedVideoUrl;
        }

        const reframedVideo = document.getElementById('reframed-video');
        reframedVideo.src = outputUrl;

        // Set up download link
        const downloadLink = document.getElementById('download-link');
        downloadLink.href = outputUrl;
        downloadLink.download = `reframed-${Date.now()}.mp4`;

        // Display analytics
        console.log('showResults called with analytics:', analytics);
        if (analytics) {
            console.log('Displaying analytics...');
            this.displayAnalytics(analytics);
        } else {
            console.log('No analytics to display');
        }

        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
        }
    }

    displayAnalytics(analytics) {
        const analyticsContent = document.getElementById('analytics-content');

        const analyticsHtml = `
            <div class="row">
                <div class="col-md-6">
                    <div class="analytics-stat">
                        <strong>Input Duration:</strong> ${analytics.input_duration?.toFixed(1) || 'N/A'}s
                    </div>
                    <div class="analytics-stat">
                        <strong>Resolution Change:</strong><br>
                        ${analytics.input_resolution} → ${analytics.output_resolution}
                    </div>
                    <div class="analytics-stat">
                        <strong>Frames Analyzed:</strong> ${analytics.frames_analyzed || 'N/A'}
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="analytics-stat">
                        <strong>Camera Movements:</strong> ${analytics.total_keyframes || 'N/A'}
                    </div>
                    <div class="analytics-stat">
                        <strong>Cut Transitions:</strong> ${analytics.total_cuts || 'N/A'}
                    </div>
                    <div class="analytics-stat">
                        <strong>AI Confidence:</strong> ${(analytics.average_confidence * 100)?.toFixed(1) || 'N/A'}%
                    </div>
                </div>
            </div>
            ${this.generateShotStats(analytics.subject_statistics)}
            ${this.generateShotDetectionComparison(analytics)}
        `;

        analyticsContent.innerHTML = analyticsHtml;
    }

    generateShotStats(shotStats) {
        if (!shotStats || !shotStats.shot_strategies) {
            return '<p class="text-muted">No shot statistics available.</p>';
        }

        let html = '<h6 class="mt-3 mb-2">Shot Breakdown:</h6>';

        const strategies = shotStats.shot_strategies;
        const totalShots = shotStats.total_shots || Object.values(strategies).reduce((a, b) => a + b, 0);

        Object.entries(strategies).forEach(([strategy, count]) => {
            const percentage = totalShots > 0 ? ((count / totalShots) * 100).toFixed(1) : '0';
            const strategyName = strategy.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

            html += `
                <div class="analytics-stat">
                    <strong>${strategyName}:</strong> ${count} shots (${percentage}%)
                </div>
            `;
        });

        html += `
            <div class="analytics-stat mt-2">
                <strong>Total Shots:</strong> ${totalShots}
            </div>
        `;

        return html;
    }

    generateShotDetectionComparison(analytics) {
        if (!analytics.gemini_shots && !analytics.our_keyframes) {
            return '<p class="text-muted">No shot detection data available.</p>';
        }

        let html = '<h6 class="mt-4 mb-3">🎬 Shot Detection Analysis</h6>';

        // Summary comparison
        if (analytics.shot_detection_comparison) {
            const comp = analytics.shot_detection_comparison;
            html += `
                <div class="row mb-3">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h6 class="card-title">📊 Detection Summary</h6>
                                <div class="analytics-stat"><strong>Gemini Shots:</strong> ${comp.gemini_shot_count}</div>
                                <div class="analytics-stat"><strong>Our Keyframes:</strong> ${comp.our_keyframe_count}</div>
                                <div class="analytics-stat"><strong>Cut Points:</strong> ${comp.cut_count}</div>
                                <div class="analytics-stat"><strong>Avg Shot Duration:</strong> ${comp.avg_shot_duration}s</div>
                                <div class="analytics-stat"><strong>Shortest Shot:</strong> ${comp.shortest_shot}s</div>
                                <div class="analytics-stat"><strong>Longest Shot:</strong> ${comp.longest_shot}s</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Detailed shot list (show first 15 shots with crop data)
        if (analytics.gemini_shots && analytics.gemini_shots.length > 0) {
            html += `
                <h6 class="mt-3 mb-2">🤖 Gemini AI Shot Detection & Crop Analysis (First 15)</h6>
                <div class="table-responsive" style="max-height: 400px; overflow-y: auto;">
                    <table class="table table-sm table-striped">
                        <thead class="table-dark">
                            <tr>
                                <th>#</th>
                                <th>Time Range</th>
                                <th>Duration</th>
                                <th>Crop Center</th>
                                <th>Strategy</th>
                                <th>Primary Subjects</th>
                                <th>AI Reasoning</th>
                                <th>Confidence</th>
                            </tr>
                        </thead>
                        <tbody>
            `;

            analytics.gemini_shots.slice(0, 15).forEach(shot => {
                const subjects = shot.primary_subjects && shot.primary_subjects.length > 0
                    ? shot.primary_subjects.join(', ')
                    : 'None specified';

                const centerPos = `(${shot.crop_center_x}, ${shot.crop_center_y})`;
                const timeRange = `${shot.start_time}s - ${shot.end_time}s`;

                html += `
                    <tr>
                        <td><strong>${shot.shot_number}</strong></td>
                        <td><small>${timeRange}</small></td>
                        <td>${shot.duration}s</td>
                        <td><code>${centerPos}</code></td>
                        <td><span class="badge bg-secondary">${shot.strategy.replace(/_/g, ' ')}</span></td>
                        <td><small class="text-primary">${subjects}</small></td>
                        <td><small class="text-muted">${shot.reasoning}</small></td>
                        <td><span class="badge ${shot.confidence > 0.8 ? 'bg-success' : shot.confidence > 0.6 ? 'bg-warning' : 'bg-danger'}">${(shot.confidence * 100).toFixed(0)}%</span></td>
                    </tr>
                    ${shot.description ? `
                    <tr class="table-light">
                        <td colspan="8"><small><em>📝 ${shot.description}</em></small></td>
                    </tr>
                    ` : ''}
                `;
            });

            html += `
                        </tbody>
                    </table>
                </div>
                ${analytics.gemini_shots.length > 15 ? `<small class="text-muted">Showing 15 of ${analytics.gemini_shots.length} shots</small>` : ''}

                <div class="mt-2">
                    <small class="text-muted">
                        <strong>Crop Center:</strong> (0,0) = top-left, (0.5,0.5) = center, (1,1) = bottom-right
                    </small>
                </div>
            `;
        }

        return html;
    }

    generateSubjectStats(subjectStats) {
        if (!subjectStats || Object.keys(subjectStats).length === 0) {
            return '<p class="text-muted">No subject statistics available.</p>';
        }

        let html = '<h6 class="mt-3 mb-2">Subject Statistics:</h6>';

        Object.entries(subjectStats).forEach(([subjectId, stats]) => {
            const speakingPercentage = (stats.speaking_ratio * 100).toFixed(1);
            html += `
                <div class="analytics-stat">
                    <strong>${stats.description || subjectId}:</strong><br>
                    <small>
                        Screen time: ${stats.screen_time?.toFixed(1)}s |
                        Speaking: ${speakingPercentage}% |
                        Importance: ${(stats.average_importance * 100)?.toFixed(1)}%
                    </small>
                </div>
            `;
        });

        return html;
    }

    async cancelJob() {
        if (!this.currentJobId) return;

        try {
            const response = await fetch(`/api/v1/jobs/${this.currentJobId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.showError('Job cancelled by user.');
            }
        } catch (error) {
            console.error('Cancel error:', error);
        }

        if (this.websocket) {
            this.websocket.close();
        }
    }

    showError(message) {
        this.hideAllSections();
        document.getElementById('error-section').classList.remove('d-none');
        document.getElementById('error-message').textContent = message;
        document.getElementById('process-btn').disabled = false;

        if (this.websocket) {
            this.websocket.close();
        }
    }

    hideAllSections() {
        ['progress-section', 'results-section', 'error-section'].forEach(id => {
            document.getElementById(id).classList.add('d-none');
        });
    }

    // Health monitoring methods
    startHealthMonitoring() {
        // Check immediately
        this.checkGeminiHealth();

        // Then check every 30 seconds
        this.healthCheckInterval = setInterval(() => {
            this.checkGeminiHealth();
        }, 30000);
    }

    async checkGeminiHealth() {
        try {
            const response = await fetch('/api/v1/health/gemini');
            const health = await response.json();

            this.updateHealthStatus(health);
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateHealthStatus({
                status: 'connection_failed',
                error: 'Failed to check AI status',
                response_time_ms: 0
            });
        }
    }

    updateHealthStatus(health) {
        const statusElement = document.getElementById('ai-status');
        const uploadButton = document.getElementById('process-btn');

        if (!statusElement) return;

        // Clear existing classes
        statusElement.className = 'ai-status';

        let statusText = '';
        let allowUploads = true;

        switch (health.status) {
            case 'healthy':
                statusElement.classList.add('status-healthy');
                statusText = `AI Ready (${health.response_time_ms}ms)`;
                break;

            case 'degraded':
                statusElement.classList.add('status-warning');
                statusText = `AI Degraded (${health.response_time_ms}ms)`;
                break;

            case 'model_not_found':
                statusElement.classList.add('status-error');
                statusText = 'AI Model Not Found';
                allowUploads = false;
                break;

            case 'connection_failed':
                statusElement.classList.add('status-error');
                statusText = 'AI Connection Failed';
                allowUploads = false;
                break;

            case 'quota_exceeded':
                statusElement.classList.add('status-warning');
                statusText = 'AI Quota Exceeded';
                allowUploads = false;
                break;

            default:
                statusElement.classList.add('status-unknown');
                statusText = `AI Status: ${health.status}`;
        }

        statusElement.textContent = statusText;

        // Enable/disable upload based on AI status
        if (uploadButton) {
            uploadButton.disabled = !allowUploads;
            uploadButton.title = allowUploads ? '' : 'AI service unavailable';
        }
    }

    async loadRecentJobs() {
        const recentJobsContent = document.getElementById('recent-jobs-content');

        try {
            // Show loading state
            recentJobsContent.innerHTML = `
                <div class="text-center py-4">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2 text-muted">Loading recent jobs...</p>
                </div>
            `;

            const response = await fetch('/api/v1/jobs/recent/completed');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.displayRecentJobs(data.jobs);

        } catch (error) {
            console.error('Error loading recent jobs:', error);
            recentJobsContent.innerHTML = `
                <div class="text-center py-4">
                    <i class="bi bi-exclamation-triangle text-warning fs-1"></i>
                    <p class="mt-2 text-muted">Error loading recent jobs</p>
                    <button class="btn btn-outline-primary btn-sm" onclick="app.loadRecentJobs()">
                        <i class="bi bi-arrow-clockwise"></i> Retry
                    </button>
                </div>
            `;
        }
    }

    displayRecentJobs(jobs) {
        const recentJobsContent = document.getElementById('recent-jobs-content');

        if (!jobs || jobs.length === 0) {
            recentJobsContent.innerHTML = `
                <div class="text-center py-4">
                    <i class="bi bi-inbox text-muted fs-1"></i>
                    <p class="mt-2 text-muted">No completed jobs yet. Upload a video to get started!</p>
                </div>
            `;
            return;
        }

        let html = '';
        jobs.forEach((job, index) => {
            // Format date
            const updatedDate = job.updated_at ? new Date(job.updated_at).toLocaleString() : 'Unknown';

            // Get analytics summary
            const analyticsSummary = this.getAnalyticsSummary(job.analytics);

            html += `
                <div class="job-item mb-3 p-3 border rounded">
                    <div class="row align-items-center">
                        <div class="col-md-6">
                            <h6 class="mb-1">
                                <i class="bi bi-film"></i> ${job.original_filename}
                            </h6>
                            <small class="text-muted">
                                <i class="bi bi-clock"></i> Completed: ${updatedDate}
                            </small>
                            ${analyticsSummary}
                        </div>
                        <div class="col-md-6 text-end">
                            ${job.output_url ? `
                                <a href="${job.output_url}" class="btn btn-primary btn-sm me-2" download>
                                    <i class="bi bi-download"></i> Download
                                </a>
                            ` : ''}
                            <button class="btn btn-outline-info btn-sm" onclick="app.showJobAnalytics('${job.job_id}', ${index})" data-bs-toggle="collapse" data-bs-target="#job-analytics-${index}">
                                <i class="bi bi-graph-up"></i> View Analysis
                            </button>
                        </div>
                    </div>

                    <!-- Analytics Details (Collapsible) -->
                    <div class="collapse mt-3" id="job-analytics-${index}">
                        <div class="analytics-detail-content">
                            ${job.analytics ? this.generateJobAnalytics(job.analytics) : '<p class="text-muted">No analytics available</p>'}
                        </div>
                    </div>
                </div>
            `;
        });

        recentJobsContent.innerHTML = html;
    }

    getAnalyticsSummary(analytics) {
        if (!analytics) return '';

        const stats = [];
        if (analytics.shots_analyzed) stats.push(`${analytics.shots_analyzed} shots`);
        if (analytics.total_cuts) stats.push(`${analytics.total_cuts} cuts`);
        if (analytics.input_duration) stats.push(`${Math.round(analytics.input_duration)}s duration`);

        return stats.length > 0 ? `<div><small class="text-info">${stats.join(' • ')}</small></div>` : '';
    }

    generateJobAnalytics(analytics) {
        if (!analytics) return '<p class="text-muted">No analytics available</p>';

        // Reuse the existing analytics generation methods
        let html = '<h6 class="mb-3">📊 Processing Summary</h6>';

        // Basic stats
        html += `
            <div class="row mb-3">
                <div class="col-md-6">
                    <div class="analytics-stat"><strong>Input Duration:</strong> ${analytics.input_duration || 'N/A'}s</div>
                    <div class="analytics-stat"><strong>Input Resolution:</strong> ${analytics.input_resolution || 'N/A'}</div>
                    <div class="analytics-stat"><strong>Output Resolution:</strong> ${analytics.output_resolution || 'N/A'}</div>
                </div>
                <div class="col-md-6">
                    <div class="analytics-stat"><strong>Shots Analyzed:</strong> ${analytics.shots_analyzed || 'N/A'}</div>
                    <div class="analytics-stat"><strong>Total Keyframes:</strong> ${analytics.total_keyframes || 'N/A'}</div>
                    <div class="analytics-stat"><strong>Average Confidence:</strong> ${analytics.average_confidence ? (analytics.average_confidence * 100).toFixed(1) + '%' : 'N/A'}</div>
                </div>
            </div>
        `;

        // Add shot detection comparison if available
        if (analytics.gemini_shots || analytics.shot_detection_comparison) {
            html += this.generateShotDetectionComparison(analytics);
        }

        return html;
    }

    showJobAnalytics(jobId, index) {
        // This method can be extended to load more detailed analytics if needed
        console.log(`Showing analytics for job ${jobId}`);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.videoReframer = new VideoReframer();
});

// Utility functions for better UX
document.addEventListener('DOMContentLoaded', () => {
    // Add tooltips to range inputs
    const smoothingRange = document.getElementById('smoothing');
    const paddingRange = document.getElementById('padding');

    smoothingRange.addEventListener('input', () => {
        const value = smoothingRange.value;
        let description = 'Medium smoothing';
        if (value < 0.3) description = 'Minimal smoothing - more responsive';
        else if (value > 0.7) description = 'High smoothing - very stable';

        smoothingRange.nextElementSibling.textContent = `${description} (${value})`;
    });

    paddingRange.addEventListener('input', () => {
        const value = paddingRange.value;
        let description = 'Standard padding';
        if (value < 1.2) description = 'Tight framing';
        else if (value > 1.5) description = 'Loose framing';

        paddingRange.nextElementSibling.textContent = `${description} (${value}x)`;
    });

    // Trigger initial updates
    smoothingRange.dispatchEvent(new Event('input'));
    paddingRange.dispatchEvent(new Event('input'));

    // Add form validation feedback
    const form = document.getElementById('upload-form');
    form.addEventListener('submit', (e) => {
        const fileInput = document.getElementById('video-file');
        const videoReframer = window.videoReframer;

        if (!fileInput.files[0] && !videoReframer?.selectedFile) {
            e.preventDefault();
            fileInput.classList.add('is-invalid');

            // Remove invalid class after file selection
            fileInput.addEventListener('change', () => {
                fileInput.classList.remove('is-invalid');
            }, { once: true });
        }
    });
});