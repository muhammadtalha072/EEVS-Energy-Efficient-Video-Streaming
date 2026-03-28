EEVS: Energy Efficient Video Streaming
Final Year Project
An AI-powered self-hosted video streaming platform that intelligently recommends the most battery-friendly quality and brightness settings — specially designed for OLED devices.
Tired of your phone or TV draining battery while streaming? EEVS automatically analyzes each video and suggests the optimal playback settings to deliver great viewing experience with maximum energy savings.
✨ Key Features

Upload your videos and get automatic FFmpeg DASH encoding in multiple quality levels
Real-time video analysis: extracts bitrate, luminance (brightness), resolution, and more
Smart ML model (trained on the OLED-EQ dataset) accurately predicts power consumption
Live AI recommendations with estimated energy savings percentage
Smooth adaptive playback powered by Dash.js
Reads real battery level from the device for even smarter suggestions

🛠 Tech Stack

Backend: Flask + Python
Video Streaming: FFmpeg + MPEG-DASH
Frontend Player: Dash.js
Machine Learning: Random Forest / LightGBM (achieving 99.8% accuracy)
Dataset: OLED-EQ (real energy consumption data from OLED TVs at different brightness levels)

Demo
https://www.linkedin.com/posts/muhammad-talha-0856663b9_fyp-finalyearproject-machinelearning-ugcPost-7443622020057022464-Ur1T?utm_source=share&utm_medium=member_desktop&rcm=ACoAAGYUJuUBSyn8WTtCf1vp52v9_k0YABA3v1M
