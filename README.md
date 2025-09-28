# Intelligent Video Reframer

AI-powered video reframing service that intelligently converts 16:9 horizontal videos to 9:16 vertical format using Google Gemini AI for scene understanding and intelligent subject tracking.

## Features

- **Intelligent Scene Analysis**: Uses Gemini AI to understand video content and identify key subjects
- **Smooth Camera Movement**: Generates natural camera movements with physics-based smoothing
- **Multi-Subject Tracking**: Tracks and prioritizes multiple subjects across frames
- **Scene-Aware Strategies**: Different framing strategies for interviews, vlogs, sports, presentations, etc.
- **Cut Detection**: Smart decisions on when to cut vs. pan between distant subjects
- **Text Preservation**: Ensures important text and graphics remain readable
- **Audio-Guided Focus**: Uses audio analysis to identify active speakers
- **Real-time Progress**: WebSocket support for live progress updates
- **Preview Generation**: Generate previews before full processing

## Architecture

- **FastAPI**: Async REST API with WebSocket support
- **Redis**: Job queue and real-time status tracking
- **S3**: Scalable video storage (input/output)
- **Gemini AI**: Intelligent scene understanding and subject detection
- **FFmpeg**: High-quality video processing and manipulation
- **Railway**: Cloud deployment platform

## Quick Start

### 1. Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template-id)

Or manually:

1. Fork this repository
2. Connect your GitHub repo to Railway
3. Add Redis service in Railway dashboard
4. Configure environment variables (see below)
5. Deploy

### 2. Environment Variables

Required environment variables:

```bash
# AWS S3
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET=your-video-bucket

# Google Gemini
GEMINI_API_KEY=your_gemini_api_key

# Optional Configuration
MAX_VIDEO_DURATION=600  # 10 minutes max
OUTPUT_VIDEO_QUALITY=high  # high, medium, fast
FRAME_ANALYSIS_FPS=2.0  # Frames per second to analyze
```

### 3. Usage

#### Create Reframing Job

```bash
curl -X POST https://your-app.railway.app/api/v1/reframe \
  -H "Content-Type: application/json" \
  -d '{
    "input_url": "s3://your-bucket/input-video.mp4",
    "output_bucket": "your-bucket",
    "output_key": "output/reframed-video.mp4",
    "settings": {
      "mode": "auto",
      "style": "vlog",
      "smoothing": 0.8,
      "quality": "high",
      "preserve_text": true
    },
    "gemini_prompts": {
      "custom_focus": "Keep the main speaker centered",
      "priority_subjects": ["person_speaking"]
    },
    "webhook_url": "https://your-webhook.com/callback"
  }'
```

#### Check Job Status

```bash
curl https://your-app.railway.app/api/v1/jobs/{job_id}
```

#### Real-time Progress (WebSocket)

```javascript
const ws = new WebSocket('wss://your-app.railway.app/api/v1/ws/{job_id}');

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log(`Progress: ${update.progress}% - ${update.message}`);

  if (update.status === 'completed') {
    console.log('Output URL:', update.output_url);
    console.log('Analytics:', update.analytics);
  }
};
```

## API Reference

### Endpoints

- `POST /api/v1/reframe` - Create reframing job
- `GET /api/v1/jobs/{job_id}` - Get job status
- `DELETE /api/v1/jobs/{job_id}` - Cancel job
- `GET /api/v1/jobs` - List recent jobs
- `POST /api/v1/jobs/{job_id}/retry` - Retry failed job
- `WS /api/v1/ws/{job_id}` - Real-time updates
- `GET /api/v1/health` - Health check
- `GET /api/v1/metrics` - Service metrics

### Request Parameters

#### ReframeSettings

- `mode`: Processing mode (`auto`, `face_priority`, `action`, `speaker`, `custom`)
- `style`: Video style (`documentary`, `vlog`, `sports`, `presentation`, `interview`, `music`, `auto`)
- `smoothing`: Smoothing factor (0.0-1.0, default: 0.8)
- `padding`: Subject padding (1.0-2.0, default: 1.2)
- `quality`: Output quality (`high`, `medium`, `fast`)
- `min_hold_time`: Minimum time to hold on subject (seconds)
- `cut_threshold`: Distance threshold for cuts vs pans
- `enable_cuts`: Enable cut transitions
- `preserve_text`: Preserve text readability
- `audio_analysis`: Use audio for speaker detection

#### GeminiPrompts

- `custom_focus`: Custom focusing instructions
- `exclude_areas`: Areas to avoid in framing
- `priority_subjects`: Subjects to prioritize

## Local Development

### Prerequisites

- Python 3.11+
- FFmpeg
- Redis
- AWS S3 bucket
- Google Gemini API key

### Setup

```bash
git clone https://github.com/your-username/video-reframer.git
cd video-reframer

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your credentials

# Start Redis
redis-server

# Run the application
uvicorn app.main:app --reload --port 8080
```

### Testing

```bash
# Health check
curl http://localhost:8080/api/v1/health

# Create test job
curl -X POST http://localhost:8080/api/v1/test
```

## Processing Pipeline

1. **Download**: Video downloaded from S3 to temporary storage
2. **Analysis**: Frames extracted and analyzed with Gemini AI
3. **Tracking**: Subjects tracked across frames for consistency
4. **Planning**: Smooth trajectory planned with physics constraints
5. **Processing**: Video reframed using FFmpeg with dynamic cropping
6. **Upload**: Processed video uploaded to S3
7. **Cleanup**: Temporary files removed

## Intelligent Features

### Scene-Specific Strategies

- **Interview**: Hold on speakers, quick cuts for reactions
- **Vlog**: Keep vlogger centered, follow eyeline
- **Sports**: Track ball/action, predictive framing
- **Presentation**: Balance speaker and visual content
- **Music**: Follow performers, cut on beat changes

### Smart Subject Tracking

- **Priority System**: Speaker > Active person > Main subject
- **Motion Prediction**: Anticipate movement direction
- **Temporal Consistency**: Smooth subject ID transitions
- **Multi-subject Balance**: Consider group composition

### Advanced Smoothing

- **Physics-Based**: Respect velocity and acceleration limits
- **Jerk Minimization**: Ultra-smooth camera movements
- **Adaptive Smoothing**: Adjust based on content complexity
- **Cut Detection**: Smart transitions for distant subjects

## Performance

- **Processing Speed**: ~1-3x real-time depending on complexity
- **Quality**: High-quality output with H.264 encoding
- **Scalability**: Horizontal scaling with Redis job queue
- **Memory**: Efficient frame processing with streaming
- **Storage**: Automatic cleanup of temporary files

## Limitations

- **Input Format**: 16:9 videos only
- **Duration**: Max 10 minutes (configurable)
- **File Size**: Max 500MB (configurable)
- **Languages**: Works best with English audio/text

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [API Docs](https://your-app.railway.app/docs)
- **Issues**: [GitHub Issues](https://github.com/your-username/video-reframer/issues)
- **Email**: support@your-domain.com

## Acknowledgments

- **Google Gemini**: AI-powered scene understanding
- **FFmpeg**: Video processing foundation
- **Railway**: Deployment platform
- **FastAPI**: Modern Python web framework