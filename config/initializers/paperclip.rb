if Rails.env.development? || Rails.env.test?
  Paperclip::Attachment.default_options.merge!(
      storage: :filesystem
  )

else
  Paperclip::Attachment.default_options.merge!(
      storage: :s3,
      s3_credentials: {
          bucket: S3_CONFIG['s3_bucket_name'],
          access_key_id: S3_CONFIG['access_key_id'],
          secret_access_key: S3_CONFIG['secret_access_key'],
          s3_region: S3_CONFIG['s3_region'],
          s3_permissions: S3_CONFIG['s3_permissions']
      },
      url: ':s3_domain_url',
      #url: '/resources/:class/:id/:style/:filename',
      path: 'resources/:class/:id/:style/:filename'
  )

end