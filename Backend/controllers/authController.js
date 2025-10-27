import User from '../models/User.js';
import jwt from 'jsonwebtoken';
import { sendVerificationEmail } from '../utils/emailService.js';


// Generate JWT Token
const generateToken = (userId) => {
  return jwt.sign({ userId }, process.env.JWT_SECRET, {
    expiresIn: process.env.JWT_EXPIRES_IN,
  });
};

// Register new user
export const register = async (req, res) => {
  try {
    console.log('📝 Registration attempt:', req.body);
    
    const {
      fullName,
      email,
      password,
      companyName,
      jobTitle,
      companyWebsite
    } = req.body;

    // Validation
    if (!fullName || !email || !password || !companyName || !jobTitle) {
      return res.status(400).json({
        success: false,
        message: 'All required fields must be provided'
      });
    }

    // Check if user exists
    const existingUser = await User.findOne({ email: email.toLowerCase() });
    if (existingUser) {
      return res.status(400).json({
        success: false,
        message: 'User with this email already exists'
      });
    }

    // Create user
    const user = new User({
      fullName: fullName.trim(),
      email: email.toLowerCase().trim(),
      password,
      companyName: companyName.trim(),
      jobTitle: jobTitle.trim(),
      companyWebsite: companyWebsite?.trim()
    });

    // Handle file upload
    if (req.file) {
      user.verificationDocument = {
        filename: req.file.originalname,
        path: req.file.path,
        mimetype: req.file.mimetype,
        size: req.file.size
      };
    }

    // Generate verification token and save
    const verificationToken = user.generateVerificationToken();
    await user.save();

    console.log('✅ User created:', user.email);

    // Send verification email (won't block registration if it fails)
    try {
      await sendVerificationEmail(user, verificationToken);
    } catch (emailError) {
      console.log('⚠️ Email sending failed, but user was created');
    }

    // Generate auth token
    const token = generateToken(user._id);

    res.status(201).json({
      success: true,
      message: 'Account created successfully! Please check your email for verification.',
      token,
      user: {
        id: user._id,
        fullName: user.fullName,
        email: user.email,
        companyName: user.companyName,
        jobTitle: user.jobTitle,
        isVerified: user.isVerified
      }
    });

  } catch (error) {
    console.error('❌ Registration error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Error creating account'
    });
  }
};

// Login user
export const login = async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({
        success: false,
        message: 'Email and password are required'
      });
    }

    // Find user with password
    const user = await User.findOne({ email: email.toLowerCase() }).select('+password');
    
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Invalid email or password'
      });
    }

    // Check password
    const isPasswordValid = await user.comparePassword(password);
    if (!isPasswordValid) {
      return res.status(401).json({
        success: false,
        message: 'Invalid email or password'
      });
    }

    // Generate token
    const token = generateToken(user._id);

    res.json({
      success: true,
      message: 'Login successful',
      token,
      user: {
        id: user._id,
        fullName: user.fullName,
        email: user.email,
        companyName: user.companyName,
        jobTitle: user.jobTitle,
        isVerified: user.isVerified
      }
    });

  } catch (error) {
    console.error('❌ Login error:', error);
    res.status(500).json({
      success: false,
      message: 'Error during login'
    });
  }
};

// Verify email
export const verifyEmail = async (req, res) => {
  try {
    const { token } = req.body;

    if (!token) {
      return res.status(400).json({
        success: false,
        message: 'Verification token is required'
      });
    }

    const user = await User.findOne({
      verificationToken: token,
      verificationTokenExpires: { $gt: Date.now() }
    });

    if (!user) {
      return res.status(400).json({
        success: false,
        message: 'Invalid or expired verification token'
      });
    }

    user.isVerified = true;
    user.verificationToken = undefined;
    user.verificationTokenExpires = undefined;
    await user.save();

    res.json({
      success: true,
      message: 'Email verified successfully! You can now access all features.'
    });

  } catch (error) {
    console.error('❌ Email verification error:', error);
    res.status(500).json({
      success: false,
      message: 'Error verifying email'
    });
  }
};

// Resend verification email
export const resendVerification = async (req, res) => {
  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({
        success: false,
        message: 'Email is required'
      });
    }

    const user = await User.findOne({ email: email.toLowerCase() });
    
    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found with this email'
      });
    }

    if (user.isVerified) {
      return res.status(400).json({
        success: false,
        message: 'Email is already verified'
      });
    }

    // Generate new token
    const verificationToken = user.generateVerificationToken();
    await user.save();

    // Send email
    await sendVerificationEmail(user, verificationToken);

    res.json({
      success: true,
      message: 'Verification email sent successfully!'
    });

  } catch (error) {
    console.error('❌ Resend verification error:', error);
    res.status(500).json({
      success: false,
      message: 'Error sending verification email'
    });
  }
};

// Get current user
export const getCurrentUser = async (req, res) => {
  try {
    const user = await User.findById(req.userId);
    
    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }

    res.json({
      success: true,
      user: {
        id: user._id,
        fullName: user.fullName,
        email: user.email,
        companyName: user.companyName,
        jobTitle: user.jobTitle,
        isVerified: user.isVerified
      }
    });

  } catch (error) {
    console.error('❌ Get user error:', error);
    res.status(500).json({
      success: false,
      message: 'Error fetching user data'
    });
  }
};