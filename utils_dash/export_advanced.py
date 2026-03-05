import streamlit as st
import pandas as pd
import io
from datetime import datetime
import base64

class AdvancedExportSystem:
    """Advanced export system with filtered data and multiple formats."""

    @staticmethod
    def export_filtered_csv(df: pd.DataFrame, filename: str = "analysis") -> bytes:
        """
        Export filtered dataframe to CSV.

        Args:
            df: DataFrame to export
            filename: Output filename

        Returns:
            CSV bytes
        """
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue().encode()

    @staticmethod
    def export_summary_json(summary_data: dict) -> str:
        """
        Export summary data to JSON.

        Args:
            summary_data: Summary dictionary

        Returns:
            JSON string
        """
        import json
        return json.dumps(summary_data, indent=2, default=str)

    @staticmethod
    def generate_pdf_report(
        df: pd.DataFrame,
        summary: dict,
        sentiment_dist: dict,
        emotion_dist: dict
    ) -> bytes:
        """
        Generate professional PDF report.

        Args:
            df: DataFrame
            summary: Summary data
            sentiment_dist: Sentiment distribution
            emotion_dist: Emotion distribution

        Returns:
            PDF bytes
        """
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib import colors

            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            elements = []

            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#00A699'),
                spaceAfter=30,
                alignment=1
            )

            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#00A699'),
                spaceAfter=12
            )

            elements.append(Paragraph("WhatsApp AI Analytics Dashboard Report", title_style))
            elements.append(Spacer(1, 0.2*inch))

            elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            elements.append(Spacer(1, 0.3*inch))

            elements.append(Paragraph("Executive Summary", heading_style))
            elements.append(Paragraph(summary.get('conversation_summary', 'N/A'), styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))

            elements.append(Paragraph("Key Metrics", heading_style))
            metrics_data = [
                ['Metric', 'Value'],
                ['Total Messages', str(len(df))],
                ['Unique Users', str(df['user'].nunique())],
                ['Date Range', f"{df['datetime'].min().date()} to {df['datetime'].max().date()}"],
            ]

            metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00A699')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(metrics_table)
            elements.append(Spacer(1, 0.3*inch))

            elements.append(Paragraph("Sentiment Analysis", heading_style))
            sentiment_data = [['Sentiment', 'Count', 'Percentage']]

            for sentiment, stats in sentiment_dist.items():
                sentiment_data.append([
                    sentiment,
                    str(stats.get('count', 0)),
                    f"{stats.get('percentage', 0):.2f}%"
                ])

            sentiment_table = Table(sentiment_data, colWidths=[2*inch, 2*inch, 2*inch])
            sentiment_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E85D75')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(sentiment_table)
            elements.append(Spacer(1, 0.3*inch))

            elements.append(Paragraph("Key Insights", heading_style))
            for insight in summary.get('key_insights', []):
                elements.append(Paragraph(f"• {insight}", styles['Normal']))

            doc.build(elements)

            pdf_buffer.seek(0)
            return pdf_buffer.getvalue()

        except ImportError:
            st.error("ReportLab not installed. Install with: pip install reportlab")
            return None

    @staticmethod
    def render_export_buttons(df: pd.DataFrame, summary: dict):
        """Render export options in UI."""
        st.subheader("📥 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Download CSV", use_container_width=True):
                csv_data = AdvancedExportSystem.export_filtered_csv(df)
                st.download_button(
                    label="Click to download CSV",
                    data=csv_data,
                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="csv_download"
                )

        with col2:
            if st.button("📄 Download JSON", use_container_width=True):
                json_data = AdvancedExportSystem.export_summary_json(summary)
                st.download_button(
                    label="Click to download JSON",
                    data=json_data,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="json_download"
                )

        with col3:
            if st.button("📑 Download PDF Report", use_container_width=True):
                st.info("Generating PDF...")
                sentiment_dist = {}
                emotion_dist = {}

                if 'sentiment_vader' in df.columns:
                    sentiment_counts = df['sentiment_vader'].value_counts()
                    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                        count = sentiment_counts.get(sentiment, 0)
                        sentiment_dist[sentiment] = {
                            'count': count,
                            'percentage': (count / len(df) * 100) if len(df) > 0 else 0
                        }

                if 'emotion' in df.columns:
                    emotion_counts = df['emotion'].value_counts()
                    for emotion in emotion_counts.index:
                        emotion_dist[emotion.upper()] = {
                            'count': emotion_counts[emotion],
                            'percentage': (emotion_counts[emotion] / len(df) * 100)
                        }

                pdf_data = AdvancedExportSystem.generate_pdf_report(
                    df, summary, sentiment_dist, emotion_dist
                )

                if pdf_data:
                    st.download_button(
                        label="Click to download PDF",
                        data=pdf_data,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key="pdf_download"
                    )
