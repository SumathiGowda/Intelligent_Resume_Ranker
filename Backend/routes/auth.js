import express from 'express';
import {
  register,
  login,
  verifyEmail,
  resendVerification,
  getCurrentUser
} from '../controllers/authController.js';
import { protect } from '../middleware/auth.js';
import upload from '../middleware/upload.js';

const router = express.Router();

// Public routes
router.post('/register', upload.single('verificationDocument'), register);
router.post('/login', login);
router.post('/verify-email', verifyEmail);
router.post('/resend-verification', resendVerification);

// Protected routes
router.get('/me', protect, getCurrentUser);

export default router;