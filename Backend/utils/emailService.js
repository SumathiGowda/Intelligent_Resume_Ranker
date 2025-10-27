import nodemailer from 'nodemailer';

// Create email transporter
const createTransporter = () => {
  if (!process.env.EMAIL_USER || !process.env.EMAIL_PASSWORD) {
    console.log('📧 Email not configured - using mock service');
    return null;
  }

  return nodemailer.createTransport({
    service: process.env.EMAIL_SERVICE,
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASSWORD,
    },
  });
};

// Send verification email
export const sendVerificationEmail = async (user, token) => {
  try {
    const transporter = createTransporter();
    
    if (!transporter) {
      console.log('📧 MOCK EMAIL: Verification would be sent to:', user.email);
      console.log('📧 MOCK EMAIL: Token:', token);
      console.log('📧 MOCK EMAIL: URL:', `${process.env.CLIENT_URL}/verify-email/${token}`);
      return true;
    }

    const verificationUrl = `${process.env.CLIENT_URL}/verify-email/${token}`;
    
    const mailOptions = {
      from: `"ResumeRank AI" <${process.env.EMAIL_USER}>`,
      to: user.email,
      subject: 'Verify Your Email - ResumeRank AI',
      html: `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); padding: 30px; text-align: center; color: white; border-radius: 10px 10px 0 0; }
                .content { background: #f8fafc; padding: 30px; border-radius: 0 0 10px 10px; }
                .button { background: #4F46E5; color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; display: inline-block; margin: 20px 0; font-weight: bold; }
                .footer { text-align: center; margin-top: 20px; color: #64748b; font-size: 14px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 ResumeRank AI</h1>
                <h2>Verify Your Email</h2>
            </div>
            <div class="content">
                <h3>Hello ${user.fullName},</h3>
                <p>Welcome to <strong>ResumeRank AI</strong>! We're excited to have you on board.</p>
                <p>To complete your registration, please verify your email address:</p>
                
                <div style="text-align: center;">
                    <a href="${verificationUrl}" class="button">Verify Email Address</a>
                </div>
                
                <p><strong>This link expires in 24 hours.</strong></p>
                
                <p>If the button doesn't work, copy and paste this URL:</p>
                <p style="background: #e2e8f0; padding: 10px; border-radius: 5px; word-break: break-all; font-size: 12px;">
                    ${verificationUrl}
                </p>
                
                <p>Best regards,<br>The ResumeRank AI Team</p>
            </div>
            <div class="footer">
                <p>&copy; 2024 ResumeRank AI. All rights reserved.</p>
            </div>
        </body>
        </html>
      `
    };
    
    const result = await transporter.sendMail(mailOptions);
    console.log(`✅ REAL EMAIL SENT to: ${user.email}`);
    console.log(`📧 Message ID: ${result.messageId}`);
    return true;
  } catch (error) {
    console.error('❌ Email sending failed:', error.message);
    console.log('📧 Falling back to mock service');
    console.log(`📧 MOCK: Verification URL: ${process.env.CLIENT_URL}/verify-email/${token}`);
    return true;
  }
};

// Test email configuration
export const testEmailConfig = async () => {
  try {
    const transporter = createTransporter();
    if (!transporter) {
      return { success: false, message: 'Email not configured' };
    }
    await transporter.verify();
    return { success: true, message: 'Email configuration is valid' };
  } catch (error) {
    return { success: false, message: error.message };
  }
};