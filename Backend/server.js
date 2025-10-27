import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve uploaded files
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Routes
app.use('/api/auth', (await import('./routes/auth.js')).default);

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.status(200).json({ 
    success: true,
    message: '🚀 ResumeRank Backend is running!',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found'
  });
});

// Error handler
app.use((error, req, res, next) => {
  console.error('Server Error:', error);
  res.status(500).json({
    success: false,
    message: 'Internal server error'
  });
});

// MongoDB connection
const connectDB = async () => {
  try {
    const conn = await mongoose.connect(process.env.MONGODB_URI);
    console.log(`✅ MongoDB Connected: ${conn.connection.host}`);
  } catch (error) {
    console.error('❌ Database connection error:', error.message);
    process.exit(1);
  }
};

// Start server
const PORT = process.env.PORT || 5000;

const startServer = async () => {
  try {
    await connectDB();
    
    app.listen(PORT, () => {
      console.log(`🎯 ResumeRank Backend Server Started`);
      console.log(`📍 Port: ${PORT}`);
      console.log(`🌐 Environment: ${process.env.NODE_ENV}`);
      console.log(`📧 Email Service: ${process.env.EMAIL_USER ? 'Enabled' : 'Disabled'}`);
      console.log(`🔗 Health Check: http://localhost:${PORT}/api/health`);
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
};

// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
  console.log('❌ Unhandled Rejection:', err);
  process.exit(1);
});
// Test email configuration endpoint
app.get('/api/test-email', async (req, res) => {
  try {
    const { testEmailConfig } = await import('./utils/emailService.js');
    const result = await testEmailConfig();
    res.json(result);
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Email test failed: ' + error.message
    });
  }
});

startServer();

export default app;